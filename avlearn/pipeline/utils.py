import inspect
import warnings
from typing import Any, Callable, Dict, List


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


def check_missing_args(args: Dict[str, Any], method: Callable) -> Any:
    """Check if a required argument of `method` is not given in `args`."""
    sig = inspect.signature(method).parameters
    req_params = [
        key for key in sig.keys() if sig[key].default == inspect._empty
    ]

    if 'kwargs' in args:
        for k, v in args['kwargs'].items():
            args[k] = v

    for param in req_params:
        if param not in args:
            raise ValueError(
                f"Missing required argument '{param}'"
            )
