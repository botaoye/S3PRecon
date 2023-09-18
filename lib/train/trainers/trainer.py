import time
import datetime
import torch
import os
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import kornia

from lib.config import cfg
from lib.utils.bin_mean_shift import Bin_Mean_Shift
from lib.utils.data_utils import to_cuda
from lib.utils.mesh_utils import extract_mesh, refuse, transform
from lib.utils.weight_adj import adjust_weight
from lib.utils.write_ply import writePLYFileDepth


class Trainer(object):
    def __init__(self, network):
        print('GPU ID: ', cfg.local_rank)
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
            network = DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                # find_unused_parameters=True
           )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

        self.amp = cfg.amp
        if self.amp:
            self.scaler = GradScaler()

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                #batch[k] = [b.cuda() for b in batch[k]]
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            else:
                # batch[k] = batch[k].cuda()
                batch[k] = batch[k].to(self.device)
        return batch
    
    def get_loss_weights(self, epoch):
        loss_weights = dict()

        loss_weights['rgb'] = cfg.loss.rgb_weight

        loss_weights['depth'] = cfg.loss.depth_weight
        for decay_epoch in cfg.loss.depth_weight_decay_epochs:
            if epoch >= decay_epoch:
                loss_weights['depth'] *= cfg.loss.depth_weight_decay
        if epoch >= cfg.loss.depth_loss_clamp_epoch:
            loss_weights['depth_loss_clamp'] = cfg.loss.depth_loss_clamp
        
        loss_weights['joint_start'] = epoch >= cfg.loss.joint_start
        loss_weights['joint'] = cfg.loss.joint_weight

        loss_weights['ce_cls'] = torch.tensor([cfg.loss.non_plane_weight, 1.0, 1.0])
        loss_weights['ce_cls'] = to_cuda(loss_weights['ce_cls'])

        loss_weights['ce'] = cfg.loss.ce_weight
        for decay_epoch in cfg.loss.ce_weight_decay_epochs:
            if epoch >= decay_epoch:
                loss_weights['ce'] *= cfg.loss.ce_weight_decay
        
        loss_weights['eikonal'] = cfg.loss.eikonal_weight

        loss_weights['plane_constrain_start'] = epoch >= 10
        # loss_weights['plane_constrain_start'] = (epoch in [10, 11, 12, 13, 14, 15,
        #                                                    20, 21, 22, 23, 24, 25, 26,
        #                                                    30, 31, 32, 33, 34, 35, 36, 37, 38,
        #                                                    40, 41, 42, 43, 44, 45,
        #                                                    ])
        # loss_weights['plane_constrain'] = 0.1  # 0.1
        loss_weights['plane_constrain'] = 0.1  # 0.1

        # loss_weights['sim_thr'] = adjust_weight(epoch, start_epoch=10, total_epochs=40, base_weight=0.5, max_weight=0.9)

        loss_weights['normal_l1'] = cfg.loss.normal_l1_weight
        loss_weights['normal_cos'] = cfg.loss.normal_cos_weight

        return loss_weights

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        loss_weights = self.get_loss_weights(epoch)

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = to_cuda(batch, self.device)
            batch['loss_weights'] = loss_weights

            optimizer.zero_grad()

            if self.amp:
                with autocast():
                    output, loss, loss_stats, image_stats = self.network(batch)
                    loss = loss.mean()
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 40)

                self.scaler.step(optimizer)
                self.scaler.update()

            else:
                output, loss, loss_stats, image_stats = self.network(batch)

                # training stage: loss; optimizer; scheduler
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
                optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, save_mesh=True, evaluate_mesh=False, data_loader=None, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        mesh = extract_mesh(self.network.net.model.sdf_net)
        if save_mesh and not evaluate_mesh:
            os.makedirs(f'{cfg.result_dir}/', exist_ok=True)
            mesh.export(f'{cfg.result_dir}/{epoch}.obj')
        if evaluate_mesh:
            assert data_loader is not None
            assert evaluator is not None
            mesh = refuse(mesh, data_loader)
            mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)
            mesh_gt = o3d.io.read_triangle_mesh(f'{cfg.test_dataset.data_root}/{cfg.test_dataset.scene}/gt.obj')
            evaluate_result = evaluator.evaluate(mesh, mesh_gt)
            print(evaluate_result)

    def render(self, epoch, data_loader=None, save_dir='', return_sep_plane=False):
        import cv2
        import numpy as np
        from lib.utils.bin_mean_shift import gen_segmentation
        from lib.utils.bin_mean_shift_normal import Bin_Mean_Shift_Normal

        self.network.eval()
        torch.cuda.empty_cache()
        # bin_mean_shift = Bin_Mean_Shift(device=self.device, )
        bin_mean_shift = Bin_Mean_Shift_Normal(device=self.device, )

        H, W = 480, 640
        # dataset_dir = data_loader.dataset.instance_dir
        # save_dir = os.path.join(dataset_dir, 'render_plane_' + str(epoch))
        save_dir = os.path.join(save_dir, 'render_plane_' + str(epoch))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # plane_segmentation_imgs = []
        plane_masks, non_plane_masks = [], []
        plane_masks_sp, non_plane_masks_sp = [], []
        with torch.no_grad():
            for iteration, batch in enumerate(tqdm(data_loader, desc='Rendering')):
                file_name = batch['meta']['filename'][0]
                rgb = batch['rgb']  # b, hw, 3
                rgb = rgb.reshape(1, H, W, 3).permute(0, 3, 1, 2)  # b, 3, hw, 3

                batch = to_cuda(batch, self.device)
                ret = self.network.net(batch, use_surface_render=True)

                normals = ret['normals_volume']
                depths = ret['depths']
                # normals = F.normalize(normals, dim=-1)
                # normals = extras['surface_normals']
                plane_embedding = normals.reshape(H, W, 3).permute(2, 0, 1)
                # plane_embedding = ret['plane_embedding'].reshape(H, W, 2).permute(2, 0, 1)

                # calculate dissimilarity_map, which servers as plane probalility
                dis_map = dissimilarity_map(rgb.to(plane_embedding.device), plane_embedding.unsqueeze(0))
                # dis_map = dis_map.unsqueeze(0).to(torch.float)
                # dis_map = kornia.morphology.closing(dis_map, kernel=torch.ones(5, 5).to(dis_map.device)).squeeze(0)
                prob = 1.0 - dis_map.squeeze(0)

                # if 'plane_embedding' in extras:
                # prob = torch.ones_like(plane_embedding[0:1, ...])
                # TODO: tune thr, current used 0.92
                mask_threshold = 0.9
                if return_sep_plane:
                    plane_mask, non_plane_mask, seg_vis_img, plane_mask_sp, non_plane_mask_sp = gen_segmentation(bin_mean_shift, prob.reshape(H, W, 1).permute(2, 0, 1), plane_embedding, mask_threshold=mask_threshold, return_sep_plane=True)
                    plane_masks.append(plane_mask)
                    non_plane_masks.append(non_plane_mask)
                    plane_masks_sp.append(plane_mask_sp)
                    non_plane_masks_sp.append(non_plane_mask_sp)
                else:
                    plane_mask, non_plane_mask, seg_vis_img = gen_segmentation(bin_mean_shift, prob.reshape(H, W, 1).permute(2, 0, 1), plane_embedding, mask_threshold=mask_threshold)
                    plane_masks.append(plane_mask)
                    non_plane_masks.append(non_plane_mask)

                # # generate 3D plane visualization
                # plane_mask_finial = np.zeros((H, W), dtype=np.uint8)
                # for plane_idx in range(plane_mask.shape[0]):
                #     plane_i = plane_mask[plane_idx]
                #     plane_i = plane_i.astype(np.uint8) * (plane_idx + 1)
                #     plane_mask_finial += plane_i
                # writePLYFileDepth(save_dir, iteration * 10, ret['depths'].reshape(H, W).cpu().numpy(), plane_mask_finial)

                # dis_map = dis_map.unsqueeze(0).to(torch.float)
                # dis_map = erode(dis_map, ksize=3).squeeze(0)
                # # dis_map = kornia.morphology.dilation(dis_map, kernel=torch.ones(3, 3).to(dis_map.device)).squeeze(0)
                # dis_map = dis_map < (1 - mask_threshold)
                # dis_map = dis_map.permute(1, 2, 0).detach().cpu().numpy()
                # dis_map = (dis_map * 255.).astype(np.uint8)
                dis_map = dis_map < (1 - mask_threshold)
                dis_map = dis_map.permute(1, 2, 0).detach().cpu().numpy()
                dis_map = (dis_map * 255.).astype(np.uint8)
                dis_map = cv2.cvtColor(dis_map, cv2.COLOR_GRAY2BGR)

                # print(os.path.join(save_dir, file_name.replace('.png', '.jpg')))
                # cv2.imwrite(os.path.join(save_dir, file_name.replace('.png', '.jpg')), seg_vis_img[:, :, ::-1])
                normals = normals.data.cpu().reshape(H, W, 3).numpy()
                normals = normals / 2. + 0.5
                normals = (normals * 255.).astype(np.uint8)

                depth = depths.data.cpu().reshape(H, W, 1).numpy()
                depth = depth / depth.max()
                depth = (depth * 255.).astype(np.uint8)
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

                save_img = np.concatenate([rgb[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255,
                                           seg_vis_img[:, :, ::-1],
                                           dis_map[:, :, ::-1],
                                           normals[:, :, ::-1],
                                           depth[:, :, ::-1]], axis=1)
                cv2.imwrite(os.path.join(save_dir, file_name.replace('.png', '.jpg')), save_img)

                # cv2.imwrite(os.path.join(save_dir, 'z_normal_' + file_name.replace('.png', '.jpg')), normals[:, :, ::-1])
                # cv2.imwrite(os.path.join(save_dir, 'dis_thr_0p1_' + file_name.replace('.png', '.jpg')), dis_map[:, :, ::-1])

                torch.cuda.empty_cache()

        return plane_masks, non_plane_masks, plane_masks_sp, non_plane_masks_sp


def normalize(a):
    return (a - a.min())/(a.max() - a.min() + 1e-8)


def dissimilarity_map(rgb, aligned_norm):
    """
    inputs:
        cam_points             b, 4, H*W
        aligned_norm        b, 3, H, W
        rgb                               b, 3, H, W
    outputs:
        seg                b, 1, H, W
    """
    # TODO: remove rgb constraint
    # rgb = torch.zeros_like(rgb)  # b, 3, hw, 3
    # TODO: remove normal constraint
    # aligned_norm = torch.zeros_like(aligned_norm)  # b, 3, hw, 3

    # rgb = kornia.filters.gaussian_blur2d(rgb, (3, 3), (1.5, 1.5))
    # rgb = kornia.filters.median_blur(rgb, (9, 9))
    # rgb = kornia.filters.median_blur(rgb, (3, 3))

    pdist = nn.PairwiseDistance(p=2)
    def cal_dis(x, y):
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        return pdist(x, y)

    nei = 3
    batch_size = aligned_norm.shape[0]

    # move valid border from depth2norm neighborhood
    rgb = rgb[:, :, 2 * nei:, :-2 * nei]
    aligned_norm = aligned_norm[:, :, 2 * nei:, :-2 * nei]
    # comute cost

    # rgb = rgb.permute(0, 2, 3, 1)
    # aligned_norm = aligned_norm.permute(0, 2, 3, 1)

    rgb_down = cal_dis(rgb[:, :, 1:], rgb[:, :, :-1])
    rgb_right = cal_dis(rgb[:, :, :, 1:], rgb[:, :, :, :-1])

    rgb_down = torch.stack([normalize(rgb_down[i]) for i in range(batch_size)])
    rgb_right = torch.stack([normalize(rgb_right[i]) for i in range(batch_size)])

    norm_down = cal_dis(aligned_norm[:, :, 1:], aligned_norm[:, :, :-1])
    norm_right = cal_dis(aligned_norm[:, :, :, 1:], aligned_norm[:, :, :, :-1])

    # D_down = torch.stack([normalize(D_down[i]) for i in range(self.opt.batch_size)])
    norm_down = torch.stack([normalize(norm_down[i]) for i in range(batch_size)])

    # D_right = torch.stack([normalize(D_right[i]) for i in range(self.opt.batch_size)])
    norm_right = torch.stack([normalize(norm_right[i]) for i in range(batch_size)])

    # normD_down = D_down + norm_down
    # normD_right = D_right + norm_right
    #
    # normD_down = torch.stack([normalize(normD_down[i]) for i in range(self.opt.batch_size)])
    # normD_right = torch.stack([normalize(normD_right[i]) for i in range(self.opt.batch_size)])

    # get max from (rgb, normD)
    cost_down = torch.stack([rgb_down, norm_down])
    cost_right = torch.stack([rgb_right, norm_right])
    cost_down, _ = torch.max(cost_down, 0)
    cost_right, _ = torch.max(cost_right, 0)
    # get dissimilarity map visualization
    dst = cost_down[:, :, : -1] + cost_right[:, :-1, :]
    dst = F.pad(dst, (0, 2 * nei + 1, 2 * nei + 1, 0), "constant", 1)
    return dst

    # outputs[('seg_dst', 0, scale)] = dst
    # # felz_seg
    # cost_down_np = cost_down.detach().cpu().numpy()
    # cost_right_np = cost_right.detach().cpu().numpy()
    # segment = torch.stack([torch.from_numpy(
    #     felz_seg(normalize(cost_down_np[i]), normalize(cost_right_np[i]), 0, 0, self.opt.height - 2 * nei,
    #              self.opt.width - 2 * nei, scale=1, min_size=50)).cuda() for i in range(self.opt.batch_size)])
    # # pad the edges that were previously trimmed
    # segment += 1
    # segment = F.pad(segment, (0, 2 * nei, 2 * nei, 0), "constant", 0)
    # outputs[("disp2seg", 0, scale)] = segment


def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out
