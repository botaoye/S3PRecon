import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from lib.train.trainers import midas_loss
from lib.train.trainers.matching import HungarianMatcher


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.matcher = HungarianMatcher()

        # self.midas_loss = midas_loss.MidasLoss(alpha=0.1)

    def forward(self, batch):
        output = self.net(batch)
        if not self.net.training:
            return output

        loss_weights = batch['loss_weights']
        loss = 0
        scalar_stats = {}

        rgb_loss = F.l1_loss(batch['rgb'], output['rgb'], reduction='none').mean() # Eq.5
        scalar_stats.update({'rgb_loss': rgb_loss})
        loss += loss_weights['rgb'] * rgb_loss

        depth_colmap_mask = batch['depth_colmap'] > 0
        if depth_colmap_mask.sum() > 0:
            depth_loss = F.l1_loss(output['depth'][depth_colmap_mask], batch['depth_colmap'][depth_colmap_mask], reduction='none') # Eq.7
            if 'depth_loss_clamp' in loss_weights:
                depth_loss = depth_loss.clamp(max=loss_weights['depth_loss_clamp'])
            depth_loss = depth_loss.mean()
            scalar_stats.update({'depth_loss': depth_loss})
            loss += loss_weights['depth'] * depth_loss

        # # for monocular depth loss
        # depth_colmap_mask = batch['depth_mono'] > 0
        # reg_loss = self.midas_loss(output['depth'], batch['depth_mono'], depth_colmap_mask)
        # reg_loss = reg_loss * 0.1
        # scalar_stats.update({'depth_mono_loss': reg_loss})
        # loss += reg_loss

        # # plane segmentation loss
        # # make 0 fixed for non-plane region
        # indices = self.matcher(output, batch)
        # segmentation_criterion = torch.nn.CrossEntropyLoss()

        # surface_normals = output['surface_normals']
        surface_normals = output['normals_volume']
        surface_normals_normalized = F.normalize(surface_normals, dim=-1).clamp(-1., 1.)

        # # Smooth loss, optional
        # surface_normals_direct = output['surface_normals']
        # surface_normals_direct_normalized = F.normalize(surface_normals_direct, dim=-1).clamp(-1., 1.)
        # surface_normals_direct_n = output['surface_normals_n']
        # surface_normals_direct_n_normalized = F.normalize(surface_normals_direct_n, dim=-1).clamp(-1., 1.)
        #
        # surf_reg_loss_pts = (torch.linalg.norm(surface_normals_direct_normalized - surface_normals_direct_n_normalized, ord=2, dim=-1, keepdim=True))
        # # surf_reg_loss = (surf_reg_loss_pts*pixel_weight).mean()
        # surf_reg_loss = surf_reg_loss_pts.mean()
        # scalar_stats.update({'surf_reg_loss': surf_reg_loss})
        # loss += surf_reg_loss

        uncertainty = None
        if 'uncertainty' in output:
            uncertainty = output['uncertainty']  # b, N_rays, 1

        # plane constraint loss
        # with torch.no_grad():
        if loss_weights['plane_constrain_start']:
            # TODO current plane loss requires batch_size=1
            plane_mask = batch['plane_mask']  # b. N_rays, N_planes
            num_planes = plane_mask.shape[-1]
            loss_plane = torch.zeros_like(loss)
            num_valid_planes = 0.
            for idx in range(num_planes):
                plane_mask_i =  plane_mask[..., idx].squeeze(0)  # b. N_rays
                if plane_mask_i.sum() < 100:
                    continue
                plane_index_i = plane_mask_i.nonzero()
                norm_i = surface_normals_normalized.gather(1, plane_index_i.unsqueeze(0).expand(-1, -1, 3))  # b, N_rays_in_plane, 3
                norm_i_mean = norm_i.mean(dim=1, keepdim=True)
                norm_i_mean = F.normalize(norm_i_mean, dim=-1).clamp(-1., 1.)

                # add constraint to filter abnormal points
                cos_sim = get_cos_similarity(norm_i, norm_i_mean)
                keep_mask = cos_sim > 0.9
                # keep_mask = cos_sim > loss_weights['sim_thr']
                norm_i = norm_i[:, keep_mask.squeeze(), :]
                norm_i_mean = norm_i.mean(dim=1, keepdim=True)
                norm_i_mean = F.normalize(norm_i_mean, dim=-1).clamp(-1., 1.)

                norm_i_mean = norm_i_mean.detach()

                # abs_distance = abs(norm_i - norm_i_mean)
                # loss_plane_abs_i = abs_distance.sum(-1).mean()
                # loss_plane += loss_plane_abs_i

                distance = norm_i * norm_i_mean  # b, N_rays_in_plane, 3
                distance = distance.sum(dim=-1)
                loss_plane_i = (1 - distance)
                if uncertainty is not None:
                    uncertainty_i = uncertainty.gather(1, plane_index_i.unsqueeze(0)).squeeze(-1)  # b, N_rays_in_plane

                    # # Laplace assumption
                    # uncertainty = uncertainty.clamp(min=-10, max=10)
                    # loss_plane_i = loss_plane_i * torch.exp(- uncertainty_i) + uncertainty_i

                    # # Gaussian assumption
                    # uncertainty_i = uncertainty_i + 3  # +3 to make it positive
                    loss_plane_i = loss_plane_i / (2 * uncertainty_i ** 2) + torch.log(uncertainty_i) + 3

                loss_plane_i = loss_plane_i.mean()
                loss_plane += loss_plane_i

                num_valid_planes += 1
            if num_valid_planes >= 1:
                loss_plane = loss_plane / num_valid_planes * loss_weights['plane_constrain']
            scalar_stats.update({'plane_loss': loss_plane})
            loss += loss_plane

        nablas: torch.Tensor = output['nablas']
        _, _ind = output['visibility_weights'][..., :nablas.shape[-2]].max(dim=-1)
        nablas = torch.gather(nablas, dim=-2, index=_ind[..., None, None].repeat([*(len(nablas.shape)-1)*[1], 3]))
        eik_bounding_box = cfg.model.bounding_radius
        eikonal_points = torch.empty_like(nablas).uniform_(-eik_bounding_box, eik_bounding_box).to(nablas.device)
        _, nablas_eik, _ = self.net.model.sdf_net.forward_with_nablas(eikonal_points)
        nablas = torch.cat([nablas, nablas_eik], dim=-2)
        nablas_norm = torch.norm(nablas, dim=-1)
        eikonal_loss = F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean') # Eq.6
        scalar_stats.update({'eikonal_loss': eikonal_loss})
        loss += loss_weights['eikonal'] * eikonal_loss

        scalar_stats.update({'loss': loss})
        scalar_stats['beta'] = output['scalars']['beta']
        scalar_stats['theta'] = self.net.theta.data

        image_stats = {}

        return output, loss, scalar_stats, image_stats


def get_angular_error(normals_source, normals_target, mask = None, clip_angle_error = -1):
    '''Get angular error betwee predicted normals and ground truth normals
    Args:
        normals_source, normals_target: N*3
        mask: N*1 (optional, default: None)
    Return:
        angular_error: float
    '''
    inner = (normals_source * normals_target).sum(dim=-1,keepdim=True)
    norm_source =  torch.linalg.norm(normals_source, dim=-1, ord=2,keepdim=True)
    norm_target = torch.linalg.norm(normals_target, dim=-1, ord=2,keepdim=True)
    angles = torch.arccos(inner/((norm_source*norm_target) + 1e-6)) #.clip(-np.pi, np.pi)
    assert not torch.isnan(angles).any()
    if mask is None:
        mask = torch.ones_like(angles)
    if mask.ndim == 1:
        mask =  mask.unsqueeze(-1)
    assert angles.ndim == mask.ndim

    mask_keep_gt_normal = torch.ones_like(angles).bool()
    if clip_angle_error>0:
        mask_keep_gt_normal = angles < clip_angle_error
        # num_clip = mask_keep_gt_normal.sum()
    angular_error = F.l1_loss(angles*mask*mask_keep_gt_normal, torch.zeros_like(angles), reduction='sum') / (mask*mask_keep_gt_normal+1e-6).sum()
    return angular_error, mask_keep_gt_normal


def get_cos_similarity(normals_source, normals_target):
    '''Get cosine similarity between predicted normals and ground truth normals
    Args:
        normals_source, normals_target: N*3
    Return:
        cos_similarity: N*1
    '''
    inner = (normals_source * normals_target).sum(dim=-1,keepdim=True)
    norm_source =  torch.linalg.norm(normals_source, dim=-1, ord=2,keepdim=True)
    norm_target = torch.linalg.norm(normals_target, dim=-1, ord=2,keepdim=True)
    cos_similarity = inner/((norm_source*norm_target) + 1e-6)
    return cos_similarity