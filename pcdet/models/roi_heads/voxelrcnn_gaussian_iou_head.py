import torch
import torch.nn as nn

from ...utils import common_utils, loss_utils
from .voxelrcnn_head import VoxelRCNNHead


class VoxelRCNNGaussianIoUHead(VoxelRCNNHead):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(
            backbone_channels=backbone_channels,
            model_cfg=model_cfg,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            num_class=num_class,
        )
        pre_channel = self.model_cfg.REG_FC[-1]
        self.reg_std_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)
        self.reg_std_bn = nn.BatchNorm1d(self.box_coder.code_size * self.num_class)
        self.reg_std_relu = nn.ReLU(inplace=True)
        self.reg_std_fc1 = nn.Linear(self.box_coder.code_size * self.num_class, 64, bias=True)
        self.reg_std_bn1 = nn.BatchNorm1d(64)
        self.reg_std_relu1 = nn.ReLU(inplace=True)
        self.reg_std_fc2 = nn.Linear(64, 1, bias=True)
        self.reg_std_sigmoid2 = nn.Sigmoid()
        self.init_reg_std_layer_weights()

    def init_reg_std_layer_weights(self):
        nn.init.normal_(self.reg_std_layer.weight, mean=0, std=0.0001)
        nn.init.constant_(self.reg_std_layer.bias, 0.1)
        nn.init.normal_(self.reg_std_fc1.weight, mean=0, std=0.0001)
        nn.init.constant_(self.reg_std_fc1.bias, 0)
        nn.init.normal_(self.reg_std_fc2.weight, mean=0, std=0.0001)
        nn.init.constant_(self.reg_std_fc2.bias, 0)

    def forward(self, batch_dict):
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        pooled_features = self.roi_grid_pool(batch_dict)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)
        ori_rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        reg_fc_features = self.reg_fc_layers(shared_features)
        rcnn_reg = self.reg_pred_layer(reg_fc_features)
        rcnn_reg_std = self.reg_std_layer(reg_fc_features)

        reg_std_out_0 = self.reg_std_bn(rcnn_reg_std.clone())
        reg_std_out_1 = self.reg_std_relu(reg_std_out_0)
        reg_std_out_2 = self.reg_std_fc1(reg_std_out_1)
        reg_std_out_3 = self.reg_std_bn1(reg_std_out_2)
        reg_std_out_4 = self.reg_std_relu1(reg_std_out_3)
        reg_std_out_5 = self.reg_std_fc2(reg_std_out_4)
        reg_std_out_6 = self.reg_std_sigmoid2(reg_std_out_5)

        rcnn_cls = torch.sigmoid(ori_rcnn_cls) * reg_std_out_6
        rcnn_cls = torch.log((rcnn_cls + 1e-6) / (1 - rcnn_cls + 1e-6))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_box_std_preds = rcnn_reg_std.view(batch_box_preds.shape)
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['batch_box_std_preds'] = batch_box_std_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['rcnn_reg_std'] = rcnn_reg_std
            self.forward_ret_dict = targets_dict

        return batch_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']
        rcnn_reg_std = forward_ret_dict['rcnn_reg_std']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = reg_valid_mask > 0
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor)

            rcnn_loss_reg_src = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )

            rcnn_loss_reg_src = rcnn_loss_reg_src.view(rcnn_batch_size, -1)
            rcnn_reg_std = torch.clamp(rcnn_reg_std, min=-5.0, max=5.0)

            rcnn_loss_reg_src = (
                torch.exp(-rcnn_reg_std) * rcnn_loss_reg_src * fg_mask.unsqueeze(dim=-1).float()
            ).sum() / max(fg_sum, 1) * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']

            rcnn_loss_reg_log = (
                0.5 * rcnn_reg_std * fg_mask.unsqueeze(dim=-1).float()
            ).sum() / max(fg_sum, 1) * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']

            rcnn_loss_reg = rcnn_loss_reg_src + rcnn_loss_reg_log

            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
            tb_dict['rcnn_loss_reg_src'] = rcnn_loss_reg_src.item()
            tb_dict['rcnn_loss_reg_log'] = rcnn_loss_reg_log.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(rcnn_boxes3d.unsqueeze(dim=1), roi_ry).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict