import os
from functools import reduce
from typing import Any, Callable, Union

import torch
from torch.utils.data import Dataset as TorchDataset

from mmcv import Config
from datasets.mmdet3d_datasets.builder import \
    build_dataset as build_mmdet3d_dataset
from datasets.mmdet3d_datasets.nuscenes_dataset import \
    NuScenesDataset as MMDET3D_NuScenesDataset


def _rgetattr(obj, attr, *args):
    """See 
    https://stackoverflow.com/questions/31174295/ \
        getattr-and-setattr-on-nested-objects"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))


class Dataset(TorchDataset):
    """
    Dataset wrapper class for generic mmdetection3d datasets.
    """

    def __init__(self, cfg: Union[str, Config], mode: str = 'train',
                 past_timesteps: int = 0, future_timesteps: int = 0) -> None:
        """
        :param cfg: Config dict or path to config file. 
                    Config dict should at least contain the key "type".
        :param mode: Whether the dataset contains training, test, 
                     or validation data. Defaults to 'train'.
        :param past_timesteps: The number of samples preceding <index> to 
                               return per __getitem__ call. Defaults to 0.
        :param future_timesteps: The number of samples following <index> to 
                                 return per __getitem__ call. Defaults to 0.
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

        mode_cfg = _rgetattr(cfg, f'data.{mode}')  # raises AttributeError
        # if cfg does not contain mode
        self._dataset = build_mmdet3d_dataset(mode_cfg, dict(
            test_mode=mode == 'val'))  # TODO: test rigorously

        # call setters
        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: handle out of bounds indices
        indices = range(index - self.past_timesteps,
                        index + self.future_timesteps + 1)
        print(indices)
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
                raise AttributeError(f"'{type(self)}' and  \
                    '{type(self._dataset)}'"
                                     f"objects have no attribute '{name}'")

    # @staticmethod
    def _validate_timesteps_arg(func: Callable) -> Any:
        def wrapped(self, timesteps: int):
            if not isinstance(timesteps, int):
                raise TypeError(
                    f"timesteps must be int, but got {type(timesteps)}")
            if not timesteps >= 0:
                raise ValueError("Absolute value of 'timesteps' must be greater"
                                 f"than or equal to 0, but is {abs(timesteps)}")
            return func(self, timesteps)
        return wrapped

    @property
    def dataset(self):
        return self._dataset

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def past_timesteps(self) -> int:
        return self._past_timesteps

    @past_timesteps.setter
    @_validate_timesteps_arg
    def past_timesteps(self, timesteps: int) -> None:
        """
        Sets the number of samples preceding <index> to 
        return per __getitem__ call.
        """
        self._past_timesteps = timesteps

    @property
    def future_timesteps(self) -> int:
        return self._future_timesteps

    @future_timesteps.setter
    @_validate_timesteps_arg
    def future_timesteps(self, timesteps: int) -> None:
        """
        Sets the number of samples following <index> to 
        return per __getitem__ call.
        """
        self._future_timesteps = timesteps


class NuScenesDataset(Dataset):
    """
    Dataset wrapper class for mmdetection3d NuScenesDataset.
    """
    # TODO: add default cfg filepath

    def __init__(self, cfg: Union[str, Config] = "", mode: str = 'train',
                 past_timesteps: int = 0, future_timesteps: int = 0) -> None:
        """
        :param cfg: Config dict or path to config file. 
                    Config dict should at least contain the key "type".
        :param mode: Whether the dataset contains training, test, 
                     or validation data. Defaults to 'train'.
        :param past_timesteps: The number of samples preceding <index> to 
                               return per __getitem__ call. Defaults to 0.
        :param future_timesteps: The number of samples following <index> to 
                                 return per __getitem__ call. Defaults to 0.
        """
        super().__init__(cfg, mode, past_timesteps, future_timesteps)
        if not isinstance(self._dataset, MMDET3D_NuScenesDataset):
            raise TypeError("dataset must be mmdetection3d.NuScenesDataset, "
                            f"but got {type(self._dataset)}")
