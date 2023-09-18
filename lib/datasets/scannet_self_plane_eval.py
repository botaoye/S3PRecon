import copy
import math
import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
from lib.config import cfg
from lib.datasets.plane_gt_process import read_plane_seg_gt_simple, read_plane_seg_gt_eval

WALL_SEMANTIC_ID = 80
FLOOR_SEMANTIC_ID = 160


class Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], kwargs['scene']
        self.instance_dir = f'{data_root}/{scene}'
        self.split = split
        assert os.path.exists(self.instance_dir)

        image_dir = '{0}/images'.format(self.instance_dir)
        self.image_list = os.listdir(image_dir)
        self.image_list.sort(key=lambda _:int(_.split('.')[0]))

        # # remove images
        # remain_list = []
        # for imgname in self.image_list:
        #     # if not '1040' in imgname and not '1060' in imgname and not '1070' in imgname:
        #     # if not '4450' in imgname and not '4460' in imgname:
        #     # if imgname == '4370.png' or imgname == '4380.png' or imgname == '4450.png':
        #     # if imgname == '820.png' or imgname == '840.png' or imgname == '880.png' or imgname == '910.png' \
        #     #         or imgname == '940.png' or imgname == '950.png' or imgname == '1080.png' or imgname == '1600.png':
        #     # if imgname in ['3950.png', '3960.png', '3970.png', '4000.png', '4010.png', '4020.png', '4080.png', '4090.png']:
        #     # if imgname in ['860.png', '870.png', '880.png', '890.png', '900.png', '910.png', '920.png',]:
        #     if imgname in ['870.png']:
        #         remain_list.append(imgname)
        # self.image_list = remain_list
        # # end remove images

        self.n_images = len(self.image_list)
        
        self.intrinsic_all = []
        self.c2w_all = []
        self.rgb_images = []

        self.semantic_deeplab = []
        self.depth_colmap = []

        self.plane_masks = []
        self.non_plane_mask = []
        self.plane_seg = []
        self.num_planes = []
        self.depth_list = []
        self.rcnn_depth_list = []
        self.rcnn_gt_depth_list = []

        self.plane_masks_planercnn = []

        self.normals = []
        # self.depth_mono = []

        self.plane_masks_sp = []
        self.non_plane_mask_sp = []

        intrinsic = np.loadtxt(f'{self.instance_dir}/intrinsic.txt')[:3, :3]
        self.intrinsic = intrinsic
        self.H, self.W = 480, 640

        index = 0
        max_num_planes = 0
        for imgname in tqdm(self.image_list, desc='Loading dataset'):
            c2w = np.loadtxt(f'{self.instance_dir}/pose/{imgname[:-4]}.txt')
            self.c2w_all.append(c2w)
            self.intrinsic_all.append(intrinsic)

            rgb = cv2.imread(f'{self.instance_dir}/images/{imgname[:-4]}.png')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)
            _, self.H, self.W = rgb.shape
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(rgb)

            if self.split == 'train':
                depth_path = f'{self.instance_dir}/depth_colmap/{imgname[:-4]}.npy'
                if os.path.exists(depth_path):
                    depth_colmap = np.load(depth_path)
                    depth_colmap[depth_colmap > 2.0] = 0
                else:
                    depth_colmap = np.zeros((self.H, self.W), np.float32)
                depth_colmap = depth_colmap.reshape(-1)
                self.depth_colmap.append(depth_colmap)

            if self.split == 'render':
                plane_mask_path = f'{self.instance_dir}/segmentation/segmentation/{imgname[:-4]}.png'
                assert os.path.exists(plane_mask_path)
                if os.path.exists(plane_mask_path):
                    # plane_mask = np.load(plane_mask_path)
                    plane_seg = cv2.imread(plane_mask_path, -1)
                    plane_seg[plane_seg == np.max(plane_seg)] = 20  # non-planar set to 20
                    plane_mask, plane_num = read_plane_seg_gt_eval(plane_mask_path)
                else:
                    plane_mask = np.zeros((1, self.H, self.W), np.float32)
                self.plane_masks.append(plane_mask.reshape(plane_mask.shape[0], -1).transpose(1, 0))
                self.plane_seg.append(plane_seg.reshape(1, -1).transpose(1, 0))
                self.num_planes.append(plane_num)

                # load plane rcnn results
                plane_mask_path = f'{self.instance_dir}/plane_masks/{str(index)}_plane_masks_0.npy'
                if os.path.exists(plane_mask_path):
                    plane_mask = np.load(plane_mask_path)
                else:
                    plane_mask = np.zeros((1, self.H, self.W), np.float32)
                self.plane_masks_planercnn.append(plane_mask.reshape(plane_mask.shape[0], -1).transpose(1, 0))

                rcnn_depth_path = f'{self.instance_dir}/plane_rcnn_depth/{str(index)}_depth_0_final.npy'
                # rcnn_depth_path = f'{self.instance_dir}/depth_ori/{str(index)}_depth_0_final_ori.npy'
                # rcnn_depth_path = f'{self.instance_dir}/depth_gt/{str(index)}_gt_depth_0.npy'
                if os.path.exists(rcnn_depth_path):
                    rcnn_depth = np.load(rcnn_depth_path)
                else:
                    rcnn_depth = np.zeros((self.H, self.W), np.float32)
                self.rcnn_depth_list.append(rcnn_depth.reshape(-1))

                # rcnn_gt_depth_path = f'{self.instance_dir}/plane_rcnn_depth_gt/{str(index)}_gt_depth_0.npy'
                # if os.path.exists(rcnn_gt_depth_path):
                #     rcnn_gt_depth = np.load(rcnn_gt_depth_path)
                # else:
                #     rcnn_gt_depth = np.zeros((self.H, self.W), np.float32)
                # self.rcnn_gt_depth_list.append(rcnn_gt_depth.reshape(-1))

                # load gt depth
                if os.path.exists(f'{self.instance_dir}/depth'):
                    depth = cv2.imread(f'{self.instance_dir}/depth/{imgname[:-4]}.png', -1).astype(np.float32)
                    depth = depth / 1000.
                else:
                    depth = np.zeros((self.H, self.W), np.float32)
                depth = depth.reshape(-1)
                self.depth_list.append(depth)

            index += 1


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        c2w, intrinsic = self.c2w_all[idx], self.intrinsic_all[idx]

        ret = {'rgb': self.rgb_images[idx]}

        if self.split == 'train':
            rays = self.gen_rays(c2w, intrinsic)
            ret['rays'] = rays
            ret['depth_colmap'] = self.depth_colmap[idx]

            if self.plane_masks:
                ret['plane_mask'] = self.plane_masks[idx]

            if self.plane_masks_sp:
                ret['plane_mask_sp'] = self.plane_masks_sp[idx]

            if cfg.train.sampling_mode == 'random':
                ids = np.random.choice(len(rays), cfg.train.N_rays, replace=False)
            elif cfg.train.sampling_mode == 'patch':
                patch_size = int(math.sqrt(cfg.train.N_rays))
                index_all = np.linspace(0, len(rays) - 1, len(rays), dtype=np.int32).reshape(self.H, self.W)
                index_blocks = block_shaped(index_all, patch_size, patch_size)  # n, patch_size, patch_size
                num_blocks = index_blocks.shape[0]
                ids = np.random.choice(num_blocks, 1, replace=False)
                ids = index_blocks[ids, :, :].reshape(-1)
            elif cfg.train.sampling_mode == 'weight_sample':
                prob = np.ones(len(rays))
                non_plane_mask = self.non_plane_mask[idx].squeeze()
                prob[non_plane_mask] += 1
                prob = prob / prob.sum()
                ids = np.random.choice(len(rays), cfg.train.N_rays, replace=False, p=prob)
            else:
                raise NotImplementedError

            for k in ret:
                ret[k] = ret[k][ids]

        elif self.split == 'render':
            rays = self.gen_rays(c2w, intrinsic)
            ret['rays'] = rays
            ret['plane_mask'] = self.plane_masks[idx]
            ret['plane_seg'] = self.plane_seg[idx]
            ret['plane_mask_rcnn'] = self.plane_masks_planercnn[idx]
            ret['num_planes'] = self.num_planes[idx]

            ret['rcnn_depth'] = self.rcnn_depth_list[idx]
            # ret['rcnn_gt_depth'] = self.rcnn_gt_depth_list[idx]
            if self.depth_list:
                ret['gt_depth'] = self.depth_list[idx]

            # ret['plane_mask'] = self.plane_masks[idx]

            # ret['semantic_deeplab'] = self.semantic_deeplab[idx]
            # ret['depth_colmap'] = self.depth_colmap[idx]

            # ret['plane_mask'] = self.plane_masks[idx]
            # ret['normal'] = self.normals[idx]
            # ret['depth_mono'] = self.depth_mono[idx]

        else:
            ret['c2w'] = c2w
            ret['intrinsic'] = intrinsic

        ret.update({'meta': {'h': self.H, 'w': self.W, 'filename': self.image_list[idx]}})
        return ret

    def gen_rays(self, c2w, intrinsic):
        H, W = self.H, self.W
        rays_o = c2w[:3, 3]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
        XYZ = XYZ @ np.linalg.inv(intrinsic).T
        XYZ = XYZ @ c2w[:3, :3].T
        rays_d = XYZ.reshape(-1, 3)

        rays = np.concatenate([rays_o[None].repeat(H*W, axis=0), rays_d], axis=-1)
        return rays.astype(np.float32)

    def update_rendered_plane(self, plane_masks, non_plane_masks):
        for plane_mask in plane_masks:
            # self.plane_masks.append(plane_mask.reshape(plane_mask.shape[0], -1).transpose(1, 0))
            self.plane_masks.append(plane_mask.reshape(plane_mask.shape[0], self.H*self.W).transpose(1, 0))
        for non_plane_mask in non_plane_masks:
            self.non_plane_mask.append(non_plane_mask.reshape(1, -1).transpose(1, 0))

    def update_rendered_plane_sp(self, plane_masks, non_plane_masks):
        for plane_mask in plane_masks:
            # self.plane_masks.append(plane_mask.reshape(plane_mask.shape[0], -1).transpose(1, 0))
            self.plane_masks_sp.append(plane_mask.reshape(plane_mask.shape[0], self.H*self.W).transpose(1, 0))
        for non_plane_mask in non_plane_masks:
            self.non_plane_mask_sp.append(non_plane_mask.reshape(1, -1).transpose(1, 0))


def block_shaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def get_world_normal(normal, extrin):
    '''
    Args:
        normal: N*3
        extrinsics: 4*4, camera to world
    Return:
        normal: N*3, in world space
    '''
    extrinsics = copy.deepcopy(extrin)
    if torch.is_tensor(extrinsics):
        extrinsics = extrinsics.cpu().numpy()

    assert extrinsics.shape[0] == 4
    normal = normal.transpose()
    extrinsics[:3, 3] = np.zeros(3)  # only rotation, no translation

    normal_world = np.matmul(extrinsics,
                             np.vstack((normal, np.ones((1, normal.shape[1])))))[:3]
    normal_world = normal_world.transpose((1, 0))

    return normal_world.clip(-1, 1)
