import os
from functools import reduce
from typing import Any, Callable, List, Union

from mmcv import Config
from torch.utils.data import Dataset

from mmdet3d_datasets.builder import build_dataset as build_mmdet3d_dataset


def _rgetattr(obj, attr, *args):
    """See 
    https://stackoverflow.com/questions/31174295/ \
        getattr-and-setattr-on-nested-objects"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))


class DetectionDataset(Dataset):
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
        self.cfg = cfg

        if not mode in ['train', 'test', 'val']:
            raise ValueError("mode must be one of 'train', 'test' "
                             f"or 'val', but is '{mode}'")

        mode_cfg = _rgetattr(cfg, f'data.{mode}')  # raises AttributeError
        # if cfg does not contain mode
        self._dataset = build_mmdet3d_dataset(mode_cfg, dict(
            test_mode=mode == 'test'))

        # call setters
        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> list:
        start = index - self.past_timesteps
        end = index + self.future_timesteps + 1
        if start < 0 or end > len(self) - 1:
            return []  # index out of range
        return [self._dataset[i] for i in range(start, end)]
    
    def format_results(self, results: List[dict], **kwargs):
        if self._is_dataset_wrapper:
            return self._dataset.dataset.format_results(results, **kwargs)
        return self._dataset.format_results(results, **kwargs)
    
    @property
    def _is_dataset_wrapper(self) -> bool:
        """
        Checks whether self._dataset is a 
        mmdet or mmdet3d dataset wrapper.
        """
        import inspect
        
        from mmdet3d_datasets import dataset_wrappers as mmdet3d_dataset_wrappers
        from mmdet.datasets import dataset_wrappers as mmdet_dataset_wrappers
        
        def _get_module_classes(module) -> list:
            assert inspect.ismodule(module)
            return [obj for _, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__]            
            
        wrappers = _get_module_classes(mmdet3d_dataset_wrappers)
        wrappers.extend(_get_module_classes(mmdet_dataset_wrappers))
        
        for wrapper in wrappers:
            if isinstance(self._dataset, wrapper):
                return True
        return False
    
    @property
    def test_mode(self) -> bool:
        if self._is_dataset_wrapper:
            return self._dataset.dataset.test_mode
        return self._dataset.test_mode    
    
    def _validate_timesteps_arg(func: Callable) -> Any:
        def wrapped(self, timesteps: int):
            if not isinstance(timesteps, int):
                raise TypeError(
                    f"timesteps must be int, but got {type(timesteps)}")
            if not timesteps >= 0:
                raise ValueError(
                    "Absolute value of 'timesteps' must be greater"
                    f"than or equal to 0, but is {abs(timesteps)}")
            return func(self, timesteps)
        return wrapped

    @property
    def dataset(self):
        return self._dataset

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
