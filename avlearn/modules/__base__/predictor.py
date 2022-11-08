"""This module implements an abstract Predictor class for the prediction task.
All of the predictors in the framework inherit this class."""


class BasePredictor:
    def __init__(self) -> None:
        self.task = "prediction"
