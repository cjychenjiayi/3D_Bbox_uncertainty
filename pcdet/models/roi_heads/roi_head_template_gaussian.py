
import torch
import numpy as np
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils

class RoIHeadTemplateGaussian(RoIHeadTemplate):
    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1]) # Keep full shape first

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        # [Gaussian Mod] Extract uncertainty
        batch_box_uncertainty = None
        if batch_box_preds.shape[-1] > code_size:
            # Assume structure: [regression (code_size), uncertainty (code_size or less), ...]
            uncertainty_raw = batch_box_preds[..., code_size:]
            # Only take the first 3 dims for XYZ uncertainty
            # Assuming uncertainty output is log(sigma^2)
            # We want to return standard deviation for visualization: sigma = exp(0.5 * log_sigma_2)
            batch_box_uncertainty = torch.exp(0.5 * uncertainty_raw[..., 0:3]).view(batch_size, -1, 3)

            # Trim box preds to code_size for decoding
            batch_box_preds_base = batch_box_preds[..., :code_size].contiguous()
            batch_box_preds = self.box_coder.decode_torch(batch_box_preds_base, local_rois).view(-1, code_size)
        else:
            batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)

        return batch_cls_preds, batch_box_preds, batch_box_uncertainty
