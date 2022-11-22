# AV-Learn
In this project we are developing an open-source framework, that implements an end-to-end autonomous vehicle pipeline. Our pipeline currently supports the autonomous vehicle tasks of detection, tracking, and prediction, yet we leave it for future work to incorporate more networks, tasks, or even autonmous driving architecture paradigms.

## Problem Statement
Autonomous driving is traditionally perceived as a classical robotics problem, where distinct subtasks -- detection, tracking, prediction, and planning -- are executed sequentially. The detection task aims to transform sensory data into object estimates, typically in the form of 2D or 3D bounding boxes. Tracking associates and follows bounding boxes over time. Prediction anticipates future agent behaviour. And finally, planning produces trajectories for the ego-agent to follow. Notwithstanding the plethora of relevant literature on detection, tracking, prediction, and planning, most of these research studies have focused on one of these *Autonomous Vehicle (AV)* tasks in isolation, without evaluating their impact on the overall AV objective, which is to provide lateral or longitudinal control of the ego-agent. That being said, this phenomenon is mainly caused by the lack of a unified framework that would support training and evaluation of an AV pipeline as a whole. The present circumstances require from the prospective researchers to manually incorporate networks for the remaining AV tasks, thus adding a complexity overhead to their work. Therefore, it becomes evident that a framework incorporating all of the subtasks of the AV pipeline as separate modules, would facilitate relevant research on the field and simplify the entire pipeline.

Moreover, a recent surge in methods that erode the boundaries between the AV subtasks challenge the predominance of the traditional **sequential** AV paradigm. Decomposing the problem into independent modules that are executed sequentially prevents downstream tasks from correcting mistakes made by upstream ones. To this end, **multitask** methods use a joint model to solve tasks simultaneously, combining all output losses in one common objective and jointly optimizing them. On the other hand, **blackbox** methods avoid simple intermediate representations entirely. Such networks are capable of training directly from the data without the need of manual interventions. The motivation behind these techniques lies in the fact that human intuition about what sequence of tasks is necessary for autonomous driving, does not necessarily translate into optimal system performance. Nevertheless, the blackbox nature of such systems raises certain questions on their ethical and safety implications, especially since these methods have not yet been thoroughly studied or benchmarked. By designing a framework that supports sequential, multitask and blackbox training, direct comparison between these approaches will be made possible. An overview of these different AV paradigms is shown in Figure 1.


<p align="center">
<img src="docs\resources\pipeline_architectures.jpg" width="80%" alt="pipeline architectures overview">
</p>
<p align="center">Figure 1: Overview of different AV pipeline architectures.</p>


## Motivation
The main motivation for this project is to devise and implement a modular framework for combining different AV tasks and promoting research on autonomous driving. The aforementioned framework is to be of a *“plug and play”* nature, thus allowing users to quickly switch out between modules and compare different network combinations. Another focus of the project lies in allowing for sequential, multi-task, and blackbox training of the different networks, so that the impact of each module on the rest can be studied, when trained in parallel. The release of the framework as an open-source code with a focus on user-friendliness and documentation is meant to motivate the formation of an active community, centered around AV learning.

## Framework Architecture
<Replace with actual code instead of pseudocode>
The main objective of the framework is to facilitate research. The supported features should simplify the process of training and evaluating different AV sub-modules in a pipeline. The following algorithm showcases a use case of the pipeline. Upon defining the dataset to be used, and the corresponding models for detection, tracking and prediction, the user is able to train, evaluate, and visualize the results of the pipeline with three simple lines of code.  The initialization of the pipeline should also support keyword arguments to enable multitask or end-to-end training, among others. In an ideal solution, the exact same algorithm should be able to run even if a different tracker, predictor, or dataset were used. 

```
dataset = NuScenesDataset()

detector = CenterPointDetector()
tracker = CenterPointTracker()
predictor = LaneGCNPredictor()

pipeline = Pipeline([detector, tracker, predictor], **kwargs)
pipeline.train()
pipeline.evaluate()
pipeline.visualize()
```

Figure 2 shows the UML class diagram for a high level depiction of the framework's architecture. In particular, the Pipeline class initializes three different objects of the abstract Task class (*i.e.*, a detector, tracker, and predictor). The Task class is a parent class for the Detector, Tracker, and Predictor classes, which in turn share inheritance relationships with individual models (not shown in the figure). Each model class needs a Trainer and Evaluator class object, which themselves depend on the abstract Loss and Dataset classes. These classes are also parent classes for individual loss classes (*i.e*., Detection, Tracking, Prediction Loss) or datasets respectively.

<p align="center">
    <img src="docs\resources\Class_Diagram.jpg" width="80%" alt="UML class diagram">
</p>
<p align="center">Figure 2: UML class diagram for high-level depiction of the framework's architecture.</p>


## Getting started
Please refer to [getting started](docs/getting_started.md) for installation and data preparation details.

## Training
Please refer to [training](docs/training.md) for details on how to train the different supported models as part of a pipeline.

## Inference & Evaluation
Please refer to [inference & evaluation](docs/inference.md) for details on how to perform inference on a pre-trained pipeline and evaluate its detection, tracking or prediction performance.

## Experiments
Please refer to [experiments](docs/experiments.md) for details on the research questions that were explored, by utilizing the AV-Learn framework, our experimental setup, and our findings.

## Model Zoo
Results and models are available in the [model zoo](docs/model_zoo.md).
