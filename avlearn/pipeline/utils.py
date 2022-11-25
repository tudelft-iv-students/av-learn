import copy
import inspect
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import torch

from ..modules.trackers.kalman_tracker.configs.nuscenes import \
    NUSCENES_TRACKING_CLASSES


def warn_module_replacement(prev_module: Any, module: Any, task: str) -> None:
    if prev_module is not None:
        warnings.warn(
            f"You can only specify one {task} in the pipeline. "
            f"{prev_module.__class__.__name__} has been replaced by "
            f"{module.__class__.__name__}.")


def print_pipeline_info(modules: List) -> None:
    info = f"""
    Pipeline setup:
    ---------------
        Detector: {modules[0].__class__.__name__}
        Tracker: {modules[1].__class__.__name__}
        Predictor: {modules[2].__class__.__name__}
    """
    print(info)


def check_missing_args(
        args: Dict[str, Any],
        method: Callable,
        module: Any) -> Any:
    """Check if a required argument of `method` is not given in `args`."""
    sig = inspect.signature(method).parameters
    req_params = [
        key for key in sig.keys() if sig[key].default == inspect._empty
    ]

    if 'kwargs' in args:
        for k, v in args['kwargs'].items():
            args[k] = v
    del args["kwargs"]

    for param in req_params:
        if param not in args and param != "kwargs":
            raise ValueError(
                f"Missing required argument '{param}'. Please specify "
                f"'{param}' when calling the '{method.__name__}' method "
                f"of {module.__class__.__name__}."
            )


def format_for_tracking(
        detections: dict, work_dir: Union[str, Path] = None,
        save: bool = False) -> dict:
    """Format MMDet3D detection output to be compatible with the 
    trackers' input format."""
    formatted_results = copy.deepcopy(detections)

    for _token in detections['results'].keys():
        tracking_items = detections["results"][_token]
        kept_items = []
        for item in tracking_items:
            if item["detection_name"] in NUSCENES_TRACKING_CLASSES:
                kept_items.append(item)

        formatted_results["results"][_token] = kept_items

    if save:
        if work_dir is None:
            work_dir = Path("/results/formatted_detections/")
        else:
            work_dir = Path(work_dir)

        work_dir.mkdir(parents=True, exist_ok=True)

        with open(Path(work_dir) / "detections_track_format.json", "w") as f:
            json.dump(formatted_results, f)

    return formatted_results


def cpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    :param data: input dict/list/tuple.
    :returns: the transfered data.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [cpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key: cpu(_data) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cpu().numpy().tolist()
    return data
