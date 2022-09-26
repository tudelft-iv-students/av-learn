import os
from functools import reduce
from typing import Union

from mmcv import Config

from mmdetection3d.mmdet3d.datasets.builder import build_dataset
from mmdetection3d.mmdet3d.datasets.nuscenes_dataset import \
    NuScenesDataset as MMDET3D_NuScenesDataset


def _rgetattr(obj, attr, *args):
    """See https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))

def Dataset(cfg: Union[str, Config], mode: str = "train") -> None:
    if isinstance(cfg, str):
        if not os.path.isfile(cfg):
            raise FileNotFoundError(f"config not found: {cfg}")            
        cfg = Config.fromfile(cfg)
    elif not isinstance(cfg, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(cfg)}')

    mode_cfg = _rgetattr(cfg, f'data.{mode}') # raises AttributeError if cfg does not contain task
    return build_dataset(mode_cfg)

def NuScenesDataset(cfg: Union[str, Config] = r"", mode: str = "train") -> None: # TODO: add default cfg filepath 
    dataset = Dataset(cfg, mode)
    if not isinstance(dataset, MMDET3D_NuScenesDataset):
        raise TypeError(f"dataset must be mmdet3d.NuScenesDataset, but got {type(dataset)}")
    return dataset