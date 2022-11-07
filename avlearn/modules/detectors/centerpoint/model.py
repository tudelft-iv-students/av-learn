from pathlib import Path
from typing import Union

from mmcv import Config

from avlearn.modules.detectors.mmdet3d.detector import MMDet3DDetector

DEFAULT_CONFIG = \
    "centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py"


class CenterPointDetector(MMDet3DDetector):
    def __init__(
            self,
            dataset: str = "nuscenes",
            velocity: bool = True,
            checkpoint: str = None,
            config: Union[str, Path] = None,
            device: str = None) -> None:

        if dataset != "nuscenes" and config is None:
            raise NotImplementedError(
                "You need to construct a configuration file. We currently"
                " only support the nuScenes dataset by default. Please look "
                "here (https://github.com/open-mmlab/mmdetection3d/blob/master"
                "/docs/en/tutorials/config.md) for more information.")

        if config is None:
            config = DEFAULT_CONFIG

        if not velocity:
            config = "novelo_" + config

        self.cfg_file = Path(
            __file__).parents[1] / "mmdet3d/configs/centerpoint" / config
        cfg = Config.fromfile(self.cfg_file)

        super().__init__(
            cfg=cfg,
            checkpoint=checkpoint,
            model_name="Centerpoint",
            device=device)
