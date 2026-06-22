
from .voxel_rcnn import VoxelRCNN
import torch
from ..model_utils import model_nms_utils

class VoxelRCNNGaussian(VoxelRCNN):
    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            # [Gaussian Mod] Retrieve uncertainty
            batch_box_uncertainty = None
            is_log_var = False

            if batch_dict.get('batch_box_uncertainty', None) is not None:
                batch_box_uncertainty = batch_dict['batch_box_uncertainty'][batch_mask]
            elif batch_dict.get('batch_box_std_preds', None) is not None:
                # Fallback to GLENet's native output (usually Log Variance)
                batch_box_uncertainty = batch_dict['batch_box_std_preds'][batch_mask]
                is_log_var = True

            # Original variance code (can keep or ignore)
            if batch_dict.get('batch_box_std_preds', None) is not None:
                box_preds_std = batch_dict['batch_box_std_preds'][batch_mask]
            else:
                box_preds_std = None

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                assert False, "Multi class NMS not supported in Gaussian Mod yet"
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1

                selected, selected_scores, new_boxes = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds, box_preds_std=box_preds_std,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = new_boxes

                # [Gaussian Mod] Filter uncertainty by NMS selection
                final_uncertainty = None
                if batch_box_uncertainty is not None:
                    final_uncertainty = batch_box_uncertainty[selected]
                    # Take only XYZ (first 3 dims)
                    if final_uncertainty.shape[-1] >= 3:
                        final_uncertainty = final_uncertainty[:, :3]

                    if is_log_var:
                        final_uncertainty = torch.exp(0.5 * final_uncertainty)

                if 'POST_SCORE_THRESH' in post_process_cfg:
                    mask = selected_scores > post_process_cfg.POST_SCORE_THRESH
                    final_scores = final_scores[mask]
                    final_labels = final_labels[mask]
                    mask_cpu = mask.cpu().numpy()
                    final_boxes = final_boxes[mask_cpu]
                    if final_uncertainty is not None:
                        final_uncertainty = final_uncertainty[mask_cpu]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'pred_uncertainty': final_uncertainty # [Gaussian Mod]
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict
