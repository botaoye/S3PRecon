import torch
import torch.nn as nn


class MatchSegmentation(nn.Module):
    def __init__(self):
        super(MatchSegmentation, self).__init__()

    def forward(self, segmentation, gt_instance, gt_plane_num):
        """
        greedy matching
        match segmentation with ground truth instance
        :param segmentation: tensor with size (N, K)
        :param prob: tensor with size (N, 1)
        :param gt_instance: tensor with size (21, h, w)
        :param gt_plane_num: int
        :return: a (K, 1) long tensor indicate closest ground truth instance id, start from 0
        """

        n, k = segmentation.size()
        _, h, w = gt_instance.size()
        # assert (prob.size(0) == n and h * w == n)

        # ingnore non planar region
        gt_instance = gt_instance[:gt_plane_num, :, :].view(1, -1, h * w)  # (1, gt_plane_num, h*w)

        segmentation = segmentation.t().view(k, 1, h * w)  # (k, 1, h*w)

        # calculate instance wise cross entropy matrix (K, gt_plane_num)
        gt_instance = gt_instance.type(torch.float32)

        ce_loss = - (gt_instance * torch.log(segmentation + 1e-6) +
                     (1 - gt_instance) * torch.log(1 - segmentation + 1e-6))  # (k, gt_plane_num, k*w)

        ce_loss = torch.mean(ce_loss, dim=2)  # (k, gt_plane_num)

        matching = torch.argmin(ce_loss, dim=1, keepdim=True)
        # return matching

        ce_val, matching = torch.min(ce_loss, dim=1)
        areas = segmentation.squeeze(1).sum(dim=1)

        max_index = matching.max().item() + 1
        matching_new = torch.ones_like(matching) * max_index
        unique_matches = torch.unique(matching)
        # only keep match with minimum cross entropy
        for ele in unique_matches:
            mask = matching == ele
            if mask.sum() == 1:
                matching_new[mask] = ele
                continue
            ce_val_tmp = ce_val.clone()
            ce_val_tmp[~mask] = 1000000
            min_ce_idx = torch.argmin(ce_val_tmp)
            matching_new[min_ce_idx] = ele
            # areas_tmp = areas.clone()
            # areas_tmp[~mask] = -1
            # max_area_idx = torch.argmax(areas_tmp)
            # matching_new[max_area_idx] = ele

        return matching_new.unsqueeze(-1)
