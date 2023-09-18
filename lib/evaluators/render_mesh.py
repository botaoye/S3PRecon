import os

import torch
import cv2
import numpy as np
import kornia

from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

from lib.evaluators.plane_seg import eval_plane_prediction, evaluateMasks
# from lib.train.trainers.trainer import dissimilarity_map
from lib.utils.bin_mean_shift import gen_segmentation, gen_segmentation_infer, Bin_Mean_Shift, colors
from lib.utils.bin_mean_shift_normal import Bin_Mean_Shift_Normal
from lib.utils.data_utils import to_cuda
from lib.utils.match_segmentation import MatchSegmentation
from lib.utils.optimizer.bin_mean_shift_cat import Bin_Mean_Shift_Cat
from lib.utils.write_ply import writePLYFileDepth


def render_mesh(data_loader):
    H, W = 480, 640
    for iteration, batch in enumerate(tqdm(data_loader, desc='Rendering')):
        file_name = batch['meta']['filename'][0]
        c2w = batch['c2w']
        # rgb = batch['rgb']  # b, hw, 3
        # rgb = rgb.reshape(1, H, W, 3).permute(0, 3, 1, 2)  # b, 3, h, w

        extr = np.linalg.inv(c2w)
        cam.extrinsic = extr
        ctrl.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        if not args.debug:
            rgb_mesh = vis.capture_screen_float_buffer(do_render=True)
            mesh_imgs.append(np.asarray(rgb_mesh))


def eval_render(network, data_loader, device, use_surface_render=False, save_dir='', cfg=None):
    H, W = 480, 640
    match_segmentatin = MatchSegmentation()
    pixel_recall_curve = np.zeros((13))
    plane_recall_curve = np.zeros((13, 3))
    plane_Seg_Metric = np.zeros((3))

    torch.cuda.empty_cache()

    use_plane_embedding = False
    eval_plane_rcnn = False

    if use_plane_embedding:
        bin_mean_shift = Bin_Mean_Shift(device=device, )
        # bin_mean_shift = Bin_Mean_Shift_Cat(device=device, )
    else:
        bin_mean_shift = Bin_Mean_Shift_Normal(device=device, )

    # dataset_dir = data_loader.dataset.instance_dir
    # save_dir = os.path.join(dataset_dir, 'render_plane_' + str(epoch))
    save_dir = os.path.join(save_dir, 'render_plane_' + 'eval')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # plane_segmentation_imgs = []
    plane_masks, non_plane_masks = [], []
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(data_loader, desc='Rendering')):
            file_name = batch['meta']['filename'][0]
            rgb = batch['rgb']  # b, hw, 3
            rgb = rgb.reshape(1, H, W, 3).permute(0, 3, 1, 2)  # b, 3, h, w

            instance = batch['plane_mask'].to(device)
            gt_seg = batch['plane_seg'].numpy()
            plane_rcnn_instance = batch['plane_mask_rcnn'].squeeze(0).to(device)
            plane_rcnn_non_plane_region = plane_rcnn_instance.sum(dim=1) == 0
            plane_rcnn_non_plane_region = plane_rcnn_non_plane_region.cpu().numpy()
            # semantic = batch['semantic'].to(device)
            # gt_depth = batch['depth'].to(device)
            gt_plane_num = batch['num_planes'].int()

            rcnn_depth = batch['rcnn_depth'].reshape(H, W).cpu().numpy()
            # rcnn_gt_depth = batch['rcnn_gt_depth'].reshape(H, W).cpu().numpy()
            if 'gt_depth' in batch:
                gt_depth = batch['gt_depth'].reshape(H, W).cpu().numpy()

            instance = instance.reshape(1, H, W, -1).permute(0, 3, 1, 2)  # b, num_plane, h, w
            # gt_seg = gt_seg.reshape(1, H, W, 1).transpose(0, 3, 1, 2)  # b, 1, h, w
            gt_seg = gt_seg.reshape(H, W)
            # gt_plane_num = instance.shape[1]

            if not eval_plane_rcnn:
                batch = to_cuda(batch, device)
                ret = network(batch, use_surface_render=use_surface_render)

                normals = ret['normals_volume']
                rendered_color = ret['rgb']
                rendered_color = rendered_color.reshape(H, W, 3).cpu().numpy()[:, :, ::-1] * 255

                rendered_depth = ret['depths'].reshape(H, W).cpu().numpy()
                # rendered_depth = ret['depth'].reshape(H, W).cpu().numpy()
                # rendered_depth = rendered_depth / cfg.test_dataset.scale + cfg.test_dataset.offset[-1]
                rendered_depth = rendered_depth / cfg.test_dataset.scale

                # normals = F.normalize(normals, dim=-1)
                # normals = extras['surface_normals']
                if use_plane_embedding:
                    plane_embedding = ret['plane_embedding'].reshape(H, W, 2).permute(2, 0, 1)
                    # plane_embedding_normal = normals.reshape(H, W, 3).permute(2, 0, 1)
                    # plane_embedding = torch.cat([plane_embedding, plane_embedding_normal], dim=0)

                else:
                    plane_embedding = normals.reshape(H, W, 3).permute(2, 0, 1)

                # vis_normal_distribution(normals.squeeze(0).cpu().numpy(), gt_seg.reshape(-1), os.path.join(save_dir, 'dis_' + file_name.replace('.png', '')))

                # TODO: tune thr, current used 0.92
                # mask_threshold = 0.9
                mask_threshold = 0.85

                # calculate dissimilarity_map, which servers as plane probalility
                dis_map = dissimilarity_map(rgb.to(plane_embedding.device), plane_embedding.unsqueeze(0))
                # dis_map_rgb, dis_map_normal, dst_all = dissimilarity_map_sep(rgb.to(plane_embedding.device), plane_embedding.unsqueeze(0))

                # dis_map = dis_map.unsqueeze(0).to(torch.float)
                # # dis_map = erode(dis_map, ksize=3).squeeze(0)
                # # dis_map = kornia.morphology.dilation(dis_map, kernel=torch.ones(3, 3).to(dis_map.device)).squeeze(0)
                # # dis_map = kornia.morphology.dilation(dis_map, kernel=torch.ones(5, 5).to(dis_map.device)).squeeze(0)
                # dis_map = kornia.morphology.closing(dis_map, kernel=torch.ones(3, 3).to(dis_map.device)).squeeze(0)

                # dis_map_normal = dis_map_normal.unsqueeze(0).to(torch.float)
                # dis_map_normal = kornia.morphology.closing(dis_map_normal, kernel=torch.ones(3, 3).to(dis_map_normal.device)).squeeze(0)
                # dis_map_rgb = dis_map_rgb.unsqueeze(0).to(torch.float)
                # dis_map_rgb = kornia.morphology.closing(dis_map_rgb, kernel=torch.ones(3, 3).to(dis_map_normal.device)).squeeze(0)
                #
                # prob_rgb = 1.0 - dis_map_rgb.squeeze(0)
                # prob_normal = (1.0 - dis_map_normal.squeeze(0)) - 0.12
                # prob, _ = torch.stack([prob_rgb, prob_normal], dim=0).min(0, keepdim=True)

                prob = 1.0 - dis_map.squeeze(0)

                segmentation = gen_segmentation_infer(bin_mean_shift,
                                                      prob.reshape(H, W, 1).permute(2, 0, 1),
                                                      plane_embedding,
                                                      mask_threshold=mask_threshold,
                                                      return_sep_plane=(not use_plane_embedding))

            else:
                # eval plane rcnn
                segmentation = plane_rcnn_instance

            # # sort segmentation by the area
            # segmentation_area = segmentation.sum(dim=0, keepdim=True)  # 1, k
            # index = segmentation_area.argsort(descending=True).expand_as(segmentation)
            # segmentation_new = torch.gather(segmentation, 1, index)
            # segmentation = segmentation_new

            # matching = match_segmentatin(segmentation, prob.view(-1, 1), instance[0], gt_plane_num)
            matching = match_segmentatin(segmentation, instance[0], gt_plane_num)

            # return cluster results
            predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

            predNumPlanes = len(np.unique(predict_segmentation)) - 1
            if predNumPlanes > 20:
                print('predNumPlanes > 20')

            # reindexing to matching gt segmentation for better visualization
            matching = matching.cpu().numpy().reshape(-1)
            used = set([])
            max_index = max(matching) + 1
            for i, a in zip(range(len(matching)), matching):
                if a in used:
                    matching[i] = max_index
                    max_index += 1
                else:
                    used.add(a)
            predict_segmentation = matching[predict_segmentation]

            non_plane_index = 20

            if eval_plane_rcnn:
                # for plane rcnn
                predict_segmentation[plane_rcnn_non_plane_region] = non_plane_index
            else:
                # mask out non planar region
                predict_segmentation[prob.cpu().numpy().reshape(-1) <= mask_threshold] = non_plane_index
            predict_segmentation = predict_segmentation.reshape(H, W)

            # generate 3D plane visualization
            # plane_mask_finial = np.zeros((H, W), dtype=np.uint8)
            # for plane_idx in range(segmentation.shape[0]):
            #     plane_i = segmentation[plane_idx]
            #     plane_i = plane_i.astype(np.uint8) * (plane_idx + 1)
            #     plane_mask_finial += plane_i
            # writePLYFileDepth(save_dir, iteration * 10, ret['depths'].reshape(H, W).cpu().numpy(), plane_mask_finial)
            # writePLYFileDepth(save_dir, iteration * 10, rendered_depth, plane_mask_finial)
            writePLYFileDepth(save_dir, file_name.replace('.png', ''), rendered_depth, predict_segmentation, non_plane_index)

            # visualization and evaluation
            h, w = H, W
            image = rgb[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255
            # semantic = semantic.cpu().numpy().reshape(h, w)
            gt_seg = gt_seg.reshape(h, w)
            # depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
            # per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

            # evaluation plane segmentation
            if eval_plane_rcnn:
                pred_depth = rcnn_depth
                # gt_depth = rcnn_gt_depth
            else:
                pred_depth = rendered_depth
            pixelStatistics, planeStatistics = eval_plane_prediction(
                predict_segmentation, gt_seg, pred_depth, gt_depth)
            # pixelStatistics, planeStatistics = eval_plane_prediction(
            #     predict_segmentation, gt_seg, None, None)
            plane_Seg_Statistics = evaluateMasks(predict_segmentation, gt_seg, device, pred_non_plane_idx=20, gt_non_plane_idx=20)
            plane_Seg_Metric += np.array(plane_Seg_Statistics)

            pixel_recall_curve += np.array(pixelStatistics)
            plane_recall_curve += np.array(planeStatistics)

            # print("pixel and plane recall of test image ", iteration)
            # print(pixel_recall_curve / float(iteration+1))
            # print(plane_recall_curve[:, 0] / plane_recall_curve[:, 1])
            # # ------------------------------------ log info
            # print(f"RI(+):{plane_Seg_Statistics[0]:.3f} | VI(-):{plane_Seg_Statistics[1]:.3f} | SC(+):{plane_Seg_Statistics[2]:.3f}")
            # print("********")

            # visualization convert labels to color image
            # change non-planar regions to zero, so non-planar regions use the black color
            gt_seg += 1
            gt_seg[gt_seg == 21] = 0
            predict_segmentation += 1
            predict_segmentation[predict_segmentation == 21] = 0

            gt_seg_image = np.stack([colors[gt_seg, 0], colors[gt_seg, 1], colors[gt_seg, 2]], axis=2)
            pred_seg = np.stack([colors[predict_segmentation, 0], colors[predict_segmentation, 1], colors[predict_segmentation, 2]], axis=2)

            # blend image
            blend_pred = (pred_seg * 0.6 + image * 0.4).astype(np.uint8)
            blend_gt = (gt_seg_image * 0.6 + image * 0.4).astype(np.uint8)

            image_1 = np.concatenate((image, pred_seg, gt_seg_image), axis=1)
            image_2 = np.concatenate((image, blend_pred, blend_gt), axis=1)

            if not eval_plane_rcnn:
                mask = (prob > mask_threshold).float().cpu().numpy().reshape(h, w)
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                normals = normals.data.cpu().reshape(H, W, 3).numpy()
                normals = normals / 2. + 0.5
                normals = (normals * 255.).astype(np.uint8)
                normals = normals[:, :, ::-1]

                image_3 = np.concatenate((rendered_color, mask, normals), axis=1)
                image = np.concatenate((image_1, image_2, image_3), axis=0)
            else:
                # depth = rcnn_depth.reshape(H, W, 1)
                # depth = depth / depth.max()
                # depth = (depth * 255.).astype(np.uint8)
                # depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

                # image_3 = np.concatenate((depth, depth, depth), axis=1)
                # image = np.concatenate((image_1, image_2, image_3), axis=0)
                image = np.concatenate((image_1, image_2), axis=0)

            cv2.imwrite(os.path.join(save_dir, file_name.replace('.png', '.jpg')), image)

            cv2.imwrite(os.path.join(save_dir, 'pred_seg_' + file_name.replace('.png', '.jpg')), pred_seg)
            # cv2.imwrite(os.path.join(save_dir, 'gt_seg_' + file_name.replace('.png', '.jpg')), gt_seg_image)
            cv2.imwrite(os.path.join(save_dir, 'b_pred_seg_' + file_name.replace('.png', '.jpg')), blend_pred)
            # cv2.imwrite(os.path.join(save_dir, 'b_gt_seg_' + file_name.replace('.png', '.jpg')), blend_gt)
            cv2.imwrite(os.path.join(save_dir, 'normal_' + file_name.replace('.png', '.jpg')), normals)
            # cv2.imwrite(os.path.join(save_dir, 'rgb_' + file_name.replace('.png', '.jpg')), rendered_color)

            depth = rendered_depth.reshape(H, W, 1)
            depth = depth / depth.max()
            depth = (depth * 255.).astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(save_dir, 'depth_' + file_name.replace('.png', '.jpg')), depth)

            torch.cuda.empty_cache()

        print("========================================")
        print("pixel and plane recall of all test image")
        print(pixel_recall_curve / len(data_loader))
        print(plane_recall_curve[:, 0] / plane_recall_curve[:, 1])
        plane_Seg_Metric = plane_Seg_Metric / len(data_loader)
        print(f"RI(+):{plane_Seg_Metric[0]:.3f} | VI(-):{plane_Seg_Metric[1]:.3f} | SC(+):{plane_Seg_Metric[2]:.3f}")
        print("****************************************")

    # return plane_masks, non_plane_masks


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
    rgb = kornia.filters.median_blur(rgb, (9, 9))
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


def dissimilarity_map_sep(rgb, aligned_norm):
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
    rgb = kornia.filters.median_blur(rgb, (5, 5))
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
    # return dst
    dst_normal = norm_down[:, :, : -1] + norm_right[:, :-1, :]
    dst_normal = F.pad(dst_normal, (0, 2 * nei + 1, 2 * nei + 1, 0), "constant", 1)
    dst_rgb = rgb_down[:, :, : -1] + rgb_right[:, :-1, :]
    dst_rgb = F.pad(dst_rgb, (0, 2 * nei + 1, 2 * nei + 1, 0), "constant", 1)

    return dst_normal, dst_rgb, dst


def vis_normal_distribution(normals, plane_mask, path):
    # normals: HW, 3
    # plane_mask: HW
    # 380

    import seaborn as sns
    import numpy as np

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import ListedColormap
    import matplotlib.ticker as ticker

    # axes instance
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # get colormap from seaborn
    # cmap = ListedColormap(sns.color_palette().as_hex())
    # cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
    cmap = colors[1:][:, ::-1] / 255.0

    # get vis data
    bg_mask = plane_mask == 20
    fg_mask = plane_mask != 20

    ids = np.random.choice(fg_mask.sum(), 5000, replace=False)

    # x = normals[:, 0][fg_mask]
    # y = normals[:, 1][fg_mask]
    # z = normals[:, 2][fg_mask]
    # color = plane_mask[fg_mask]
    x = normals[:, 0][fg_mask][ids]
    y = normals[:, 1][fg_mask][ids]
    z = normals[:, 2][fg_mask][ids]
    color = plane_mask[fg_mask][ids]
    color = cmap[color]

    # np.save(path + '.npy', np.stack([x, y, z], axis=1))
    # np.save(path + '_color.npy', color)

    # plot
    # sc = ax.scatter(x, y, z, s=2, c=color, marker='o', cmap=cmap, alpha=1)
    sc = ax.scatter(x, y, z, s=2, c=color, marker='o', alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(x.min(), 0.4)
    ax.set_ylim(-0.45, y.max())
    ax.set_zlim(z.min(), 0.5)
    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # plt.axis('off')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # save
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    # plt.show()
