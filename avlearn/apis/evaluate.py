"""This module implements the evaluation of each individual task in the
AV-learn pipeline."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Union

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.tracking.data_classes import TrackingConfig
from nuscenes.eval.tracking.evaluate import TrackingEval


class Evaluator:
    """Evaluate a `task` of the AV-learn pipeline.

    This class implements the evaluation of the specified `task` for the
    corresponding `dataset`. The results are written to the given `output`
    directory. 

    In the case of NuScenes, it utilizes the official evaluation code.
    https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval
    """

    def __init__(self,
                 task: Union[str, Path],
                 dataset: Union[str, Path],
                 results: Union[str, Path],
                 output: Union[str, Path],
                 dataroot: Union[str, Path],
                 split: Union[str, Path],
                 **kwargs: Any
                 ) -> None:
        """Initialize an Evaluator.

        :param task: Task to evaluate.
        :param dataset: The dataset to evaluate on.
        :param results: Path to the JSON file with the results of the task.
        :param output: Path to the directory to save results to.
        :param dataroot: Path to the data directory.
        :param split: The dataset split to evaluate on.
        """

        self.task = task
        self.dataset = dataset
        self.results = results
        self.output = output
        self.dataroot = dataroot
        self.split = split
        self.kwargs = kwargs

        if dataset == 'nuscenes':
            self.__init_nuscenes()
        else:
            print("We currently only support the NuScenes dataset.")
            raise NotImplementedError

    def __init_nuscenes(self) -> None:
        """Initialize a NuScenes Evaluator."""
        if self.task == "detection":
            # Initialize NuScenes object
            nusc = NuScenes(
                version=self.kwargs["version"],
                verbose=self.kwargs["verbose"],
                dataroot=self.dataroot)

            if self.kwargs['config_path'] is None:
                config = config_factory('detection_cvpr_2019')
            else:
                with open(self.kwargs['config_path'], 'r') as f:
                    config = DetectionConfig.deserialize(json.load(f))

            self.evaluator = DetectionEval(
                nusc, config=config, result_path=self.results,
                eval_set=self.split, output_dir=self.output,
                verbose=self.kwargs["verbose"])

        elif self.task == 'tracking':
            if self.kwargs['config_path'] is None:
                config = config_factory('tracking_nips_2019')
            else:
                with open(self.kwargs['config_path'], 'r') as f:
                    config = TrackingConfig.deserialize(json.load(f))

            self.evaluator = TrackingEval(
                config=config, result_path=self.results, eval_set=self.split,
                output_dir=self.output, nusc_version=self.kwargs["version"],
                nusc_dataroot=self.dataroot, verbose=self.kwargs["verbose"],
                render_classes=self.kwargs['render_classes'])

        else:
            # TODO: Initialize evaluator for the prediction task
            pass

    def __init_kitti(self) -> None:
        """Initialize a KITTI Evaluator."""
        # TODO: Add support for KITTI dataset
        pass

    def __init_waymo(self) -> None:
        """Initialize a Waymo Evaluator."""
        # TODO: Add support for Waymo dataset
        pass

    def __eval_nuscenes(self) -> Dict[str, Any]:
        """Evaluate on NuScenes."""
        if self.task == 'detection':
            return self.evaluator.main(
                plot_examples=self.kwargs["plot_examples"],
                render_curves=self.kwargs["render_curves"])
        elif self.task == 'tracking':
            return self.evaluator.main(
                render_curves=self.kwargs["render_curves"])

    def __eval_kitti(self) -> Dict[str, Any]:
        """Evaluate on KITTI."""
        # TODO: Add support for KITTI dataset
        pass

    def __eval_waymo(self) -> Dict[str, Any]:
        """Evaluate on Waymo."""
        # TODO: Add support for Waymo dataset
        pass

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the `task`.

        This method evaluates the `task` on the given `dataset`, and renders
        corresponding plots. It returns the metrics computed during the 
        evaluation. 
        """
        if self.dataset == 'nuscenes':
            return self.__eval_nuscenes()
        elif self.dataset == 'kitti':
            pass
        else:  # Waymo dataset
            pass


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Evaluate the results.")

    # Main arguments
    parser.add_argument(
        '-t', "--task", type=str, help="Task to evaluate.",
        choices=["detection", "tracking", "prediction"], required=True)
    parser.add_argument('-r', "--results", type=str, required=True,
                        help="Path to the task\'s results.")
    parser.add_argument(
        "--dataset", type=str,
        help="Dataset to evaluate on.",
        required=True, choices=["nuscenes", "kitti", "waymo"])
    parser.add_argument(
        '-o', '--output', type=str, default='../nuscenes-metrics',
        help='Directory to save results to.')
    parser.add_argument(
        '--split', type=str, default='val',
        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str,
                        default='../data/nuscenes',
                        help='Default nuScenes data directory.')

    # nuscenes-devkit arguments
    parser.add_argument(
        '--version', type=str, default='v1.0-trainval',
        help='Which version of the dataset to evaluate on, e.g. v1.0-trainval')
    parser.add_argument(
        '--config_path', type=str, default='',
        help='Path to the configuration file for the evaluation.')
    parser.add_argument(
        '--plot_examples', type=int, default=10,
        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render curves to disk.')
    parser.add_argument(
        '--render_classes', type=str, default='', nargs='+',
        help='For which classes to render tracking results to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')

    args = vars(parser.parse_args())
    args['render_curves'] = bool(args['render_curves'])
    args['verbose'] = bool(args['verbose'])

    evaluator = Evaluator(**args)
    evaluator.evaluate()
