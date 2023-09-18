import cv2
import numpy as np
from scipy.ndimage.measurements import label


def read_plane_seg_gt_simple(plane_seg_filepath, planeAreaThreshold=200, seg=None):
    if seg is None:
        seg = cv2.imread(plane_seg_filepath, -1)

    if np.max(seg) >= 20:
        raise NotImplementedError

    seg[seg == np.max(seg)] = 20  # non-planar set to 20

    ori_indices, counts = np.unique(seg, return_counts=True)
    plane_num = len(ori_indices) - 1

    instance = np.zeros([plane_num, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    non_plane_mask = np.zeros([seg.shape[0], seg.shape[1]], dtype=np.uint8)

    new_index = 0
    for ori_index in ori_indices:
        if ori_index == 20:
            non_plane_mask = seg == ori_index
            assert np.sum(non_plane_mask) > 0
            continue

        if np.sum(seg == ori_index) < planeAreaThreshold:
            continue

        instance[new_index] = seg == ori_index
        new_index += 1

    return instance, non_plane_mask


def read_plane_seg_gt_eval(plane_seg_filepath, planeAreaThreshold=200, seg=None):
    if seg is None:
        seg = cv2.imread(plane_seg_filepath, -1)

    if np.max(seg) >= 20:
        raise NotImplementedError

    seg[seg == np.max(seg)] = 20  # non-planar set to 20

    ori_indices, counts = np.unique(seg, return_counts=True)
    plane_num = len(ori_indices) - 1

    instance = np.zeros([21, seg.shape[0], seg.shape[1]], dtype=np.uint8)

    new_index = 0
    for ori_index in ori_indices:
        if ori_index == 20:
            instance[20] = seg == ori_index
            assert np.sum(instance[20]) > 0
            continue

        instance[new_index] = seg == ori_index
        new_index += 1

    return instance, plane_num


def read_plane_seg_self(plane_seg_filepath, planeAreaThreshold=200, seg=None):
    if seg is None:
        seg = cv2.imread(plane_seg_filepath, -1)

    # if np.max(seg) >= 20:
    #     raise NotImplementedError

    # seg[seg == np.max(seg)] = 20  # non-planar set to 20
    ori_indices, counts = np.unique(seg, return_counts=True)
    if np.max(seg) == 20:
        plane_num = len(ori_indices) - 1
    else:
        plane_num = len(ori_indices)

    instance = np.zeros([plane_num, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    non_plane_mask = np.zeros([seg.shape[0], seg.shape[1]], dtype=np.uint8)

    new_index = 0
    for ori_index in ori_indices:
        if ori_index == 20:
            non_plane_mask = seg == ori_index
            assert np.sum(non_plane_mask) > 0
            continue

        if np.sum(seg == ori_index) < planeAreaThreshold:
            continue

        instance[new_index] = seg == ori_index
        new_index += 1

    return instance, non_plane_mask


def read_plane_seg_self_conneted(plane_seg_filepath, planeAreaThreshold=200, seg=None):
    if seg is None:
        seg = cv2.imread(plane_seg_filepath, -1)

    planeAreaThreshold = 500

    # if np.max(seg) >= 20:
    #     raise NotImplementedError

    # seg[seg == np.max(seg)] = 20  # non-planar set to 20
    # seg: 480 x 640
    # TODO: get connected component and remove small plane regions

    ori_indices, counts = np.unique(seg, return_counts=True)
    if np.max(seg) == 20:
        plane_num = len(ori_indices) - 1
    else:
        plane_num = len(ori_indices)

    # instance = np.zeros([plane_num, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    # instance = np.zeros([1, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    non_plane_mask = np.zeros([seg.shape[0], seg.shape[1]], dtype=np.uint8)
    instances = []
    structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter

    for ori_index in ori_indices:
        if ori_index == 20:
            non_plane_mask = seg == ori_index
            assert np.sum(non_plane_mask) > 0
            continue

        if np.sum(seg == ori_index) < planeAreaThreshold:
            continue

        instance_all = seg == ori_index
        labeled, ncomponents = label(instance_all, structure)
        for i in range(1, ncomponents+1):
            instance_i = labeled == i
            if np.sum(instance_i) < planeAreaThreshold:
                continue

            instance_range = np.nonzero(instance_i)
            min_row = instance_range[0].min()
            max_row = instance_range[0].max()
            min_col = instance_range[1].min()
            max_col = instance_range[1].max()
            range_row = max_row - min_row
            range_col = max_col - min_col
            if range_row < 50 or range_col < 50:
                continue

            instances.append(instance_i)

        # instance[new_index] = seg == ori_index
        # new_index += 1

    if len(instances) > 0:
        instance = np.stack(instances, axis=0)
    else:
        instance = np.zeros([0, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    return instance, non_plane_mask
