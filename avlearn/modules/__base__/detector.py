"""This module implements an abstract Detector class for the detection task.
All of the detectors in the framework inherit this class."""


class BaseDetector:
    def __init__(self) -> None:
        self.task = "detection"
