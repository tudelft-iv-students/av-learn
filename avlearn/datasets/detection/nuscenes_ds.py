from typing import Union

from avlearn.config.definitions import ROOT_DIR
from mmcv import Config

from detdataset import DetectionDataset


class NuScenesDataset(DetectionDataset):
    """
    Dataset wrapper class for mmdetection3d NuScenesDataset.
    """
    def __init__(self, cfg: Union[str, Config, None] = None, mode: str = 'train',
                 past_timesteps: int = 0, future_timesteps: int = 0) -> None:
        """
        :param cfg: Config dict or path to config file. 
                    Config dict should at least contain the key "type".
                    Defaults to None.
        :param mode: Whether the dataset contains training, test, 
                     or validation data. Defaults to 'train'.
        :param past_timesteps: The number of samples preceding <index> to 
                               return per __getitem__ call. Defaults to 0.
        :param future_timesteps: The number of samples following <index> to 
                                 return per __getitem__ call. Defaults to 0.
        """
        if cfg is None:
            cfg = str(ROOT_DIR / 'modules/detectors/mmdet3d/configs/_base_/datasets/nus-3d.py')
        super().__init__(cfg, mode, past_timesteps, future_timesteps)
        if not self.cfg.dataset_type == 'NuScenesDataset':
            raise TypeError("cfg.dataset_type must be 'NuScenesDataset', "
                            f"but got {type(self.cfg.dataset_type)}")
