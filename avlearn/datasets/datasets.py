import os
from functools import reduce
from typing import Any, Optional, Union

import torch
from torch.utils.data import Dataset as TorchDataset

from mmcv import Config
from mmdetection3d.mmdet3d.datasets.builder import \
    build_dataset as build_mmdet3d_dataset
from mmdetection3d.mmdet3d.datasets.nuscenes_dataset import \
    NuScenesDataset as MMDET3D_NuScenesDataset

def _rgetattr(obj, attr, *args):
    """See https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))


class Dataset(TorchDataset):
    """
    Dataset wrapper class for generic mmdetection3d datasets.
    """
    
    def __init__(self, cfg: Union[str, Config], mode: str = 'train', time_horizon: Optional[int] = None) -> None:
        """
        :param cfg: Config dict or path to config file. 
                    Config dict should at least contain the key "type".
        :param mode: Whether the dataset contains training, test, 
                     or validation data. Defaults to 'train'.
        :param time_horizon: (Optional) ............. Defaults to None.
        """
        if isinstance(cfg, str):
            if not os.path.isfile(cfg):
                raise FileNotFoundError(f"config not found: {cfg}")            
            cfg = Config.fromfile(cfg)
        elif not isinstance(cfg, Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(cfg)}')
            
        if not mode in ['train', 'test', 'val']:
            raise ValueError("mode must be one of 'train', 'test' " 
                             f"or 'val', but is '{mode}'")
        self._mode = mode

        mode_cfg = _rgetattr(cfg, f'data.{mode}') # raises AttributeError 
                                                  # if cfg does not contain mode                                                                          
        self._dataset = build_mmdet3d_dataset(mode_cfg, dict(test_mode=mode=='val')) # TODO: test if this works
    
        self.time_horizon = time_horizon # call setter
            
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> torch.Tensor:                               
        if self._time_horizon is None:
            indices = [index]
        elif self._time_horizon > 0:
            indices = range(index, index + self._time_horizon)
        else:
            indices = range(index - self._time_horizon, index)
        
        return torch.Tensor([self._dataset[i] for i in indices])
        
    def __getattribute__(self, name: str) -> Any:
        """
        Provides direct access to dependency injected mmdetection3d dataset. 
        :param name: Attribute name.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            try:
                return object.__getattribute__(self._dataset, name)
            except AttributeError:
                raise AttributeError(f"'{type(self)}' and '{type(self._dataset)}'"
                                     f"objects have no attribute '{name}'")
            
    @property
    def dataset(self):
        return self._dataset

    @property
    def mode(self):
        return self._mode

    @property
    def time_horizon(self):
        return self._time_horizon

    @property.setter
    def time_horizon(self, timesteps: Optional[int]) -> None: # TODO: rename tmp
        if not isinstance(timesteps, int):
            raise TypeError(f"'timesteps' must be int, but got {type(timesteps)}")
        if not abs(timesteps) >= 1:
            raise ValueError("Absolute value of 'timesteps' must be greater"
                             f"than or equal to 1, but is {abs(timesteps)}")
        self._time_horizon = timesteps
        
        
class NuScenesDataset(Dataset):
    """
    Dataset wrapper class for mmdetection3d NuScenesDataset.
    """
    def __init__(self, cfg: Union[str, Config] = "", mode: str = 'train', time_horizon: Optional[int] = None) -> None: # TODO: add default cfg filepath 
        """
        :param cfg: Config dict or path to config file. 
                    Config dict should at least contain the key "type".
        :param mode: Whether the dataset contains training, test, 
                     or validation data. Defaults to 'train'.
        :param time_horizon: (Optional) ............. Defaults to None.
        """
        super().__init__(cfg, mode)
        if not isinstance(self._dataset, MMDET3D_NuScenesDataset):
            raise TypeError("dataset must be mmdetection3d.NuScenesDataset, "
                            f"but got {type(self._dataset)}")