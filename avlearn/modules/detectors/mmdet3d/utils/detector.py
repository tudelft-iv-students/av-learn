import collections
from pathlib import Path
from typing import Union, Any

from mmcv import Config


def convert_SyncBN(config: Any) -> None:
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


def update_data_paths(input_obj: Any, dataroot: Union[str, Path]) -> None:
    """Update recursively dataroot in config file."""

    keys = ["data_root", "info_path", "ann_file"]
    if isinstance(input_obj, Config):
        for v in input_obj.values():
            update_data_paths(v, dataroot)

    if isinstance(input_obj, list):
        for item in input_obj:
            update_data_paths(item, dataroot)

    if isinstance(input_obj, collections.abc.Mapping):
        for key in keys:
            if key in input_obj:
                input_obj[key] = input_obj[key].replace(
                    "data/nuscenes", dataroot)

        for v in input_obj.values():
            update_data_paths(v, dataroot)
