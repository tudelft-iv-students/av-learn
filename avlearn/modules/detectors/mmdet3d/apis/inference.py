# Copyright (c) OpenMMLab. All rights reserved.

# Removed: All code related to segmentation
# Added: Option for init_model to return Config dict alongside model

from copy import deepcopy
from os import path as osp

import mmcv
import torch
from mmcv.runner import load_checkpoint

from avlearn.modules.detectors.mmdet3d.core import (
    Box3DMode, CameraInstance3DBoxes, Coord3DMode, DepthInstance3DBoxes,
    LiDARInstance3DBoxes, show_multi_modality_result, show_result)
from avlearn.modules.detectors.mmdet3d.models import build_model
from avlearn.modules.detectors.mmdet3d.utils import get_root_logger


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def init_model(config, checkpoint=None, device='cuda:0',
               return_config: bool = False):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    # config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    if device != 'cpu':
        torch.cuda.set_device(device)
    else:
        logger = get_root_logger()
        logger.warning('Don\'t suggest using CPU device. '
                       'Some functions are not supported for now.')
    model.to(device)
    model.eval()

    if return_config:
        return model, config
    return model


def show_det_result_meshlab(data,
                            result,
                            out_dir,
                            score_thr=0.0,
                            show=False,
                            snapshot=False):
    """Show 3D detection result by meshlab."""
    points = data['points'][0][0].cpu().numpy()
    pts_filename = data['img_metas'][0][0]['pts_filename']
    file_name = osp.split(pts_filename)[-1].split('.')[0]

    if 'pts_bbox' in result[0].keys():
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
        pred_labels = result[0]['pts_bbox']['labels_3d']
    else:
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['scores_3d'].numpy()
        pred_labels = result[0]['labels_3d']

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    # for now we convert points into depth mode
    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode != Box3DMode.DEPTH:
        points = Coord3DMode.convert(points, box_mode, Coord3DMode.DEPTH)
        show_bboxes = Box3DMode.convert(pred_bboxes, box_mode, Box3DMode.DEPTH)
    else:
        show_bboxes = deepcopy(pred_bboxes)

    show_result(
        points,
        None,
        show_bboxes,
        out_dir,
        file_name,
        show=show,
        snapshot=snapshot,
        pred_labels=pred_labels)

    return file_name


def show_proj_det_result_meshlab(data,
                                 result,
                                 out_dir,
                                 score_thr=0.0,
                                 show=False,
                                 snapshot=False):
    """Show result of projecting 3D bbox to 2D image by meshlab."""
    assert 'img' in data.keys(), 'image data is not provided for visualization'

    img_filename = data['img_metas'][0][0]['filename']
    file_name = osp.split(img_filename)[-1].split('.')[0]

    # read from file because img in data_dict has undergone pipeline transform
    img = mmcv.imread(img_filename)

    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['scores_3d'].numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode == Box3DMode.LIDAR:
        if 'lidar2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'LiDAR to image transformation matrix is not provided')

        show_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['lidar2img'],
            out_dir,
            file_name,
            box_mode='lidar',
            show=show)
    elif box_mode == Box3DMode.DEPTH:
        show_bboxes = DepthInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            None,
            out_dir,
            file_name,
            box_mode='depth',
            img_metas=data['img_metas'][0][0],
            show=show)
    elif box_mode == Box3DMode.CAM:
        if 'cam2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'camera intrinsic matrix is not provided')

        show_bboxes = CameraInstance3DBoxes(
            pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['cam2img'],
            out_dir,
            file_name,
            box_mode='camera',
            show=show)
    else:
        raise NotImplementedError(
            f'visualization of {box_mode} bbox is not supported')

    return file_name
