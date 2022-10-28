from typing import Union

from mmdet3d_datasets.nuscenes_dataset import \
    NuScenesDataset as MMDET3D_NuScenesDataset
from mmcv import Config

from detdataset import DetectionDataset


class NuScenesDataset(DetectionDataset):
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
