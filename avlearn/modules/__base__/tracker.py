"""This module implements an abstract Tracker class for the tracking task.
All of the trackers in the framework inherit this class."""


class BaseTracker:
    def __init__(self) -> None:
        self.task = "tracking"
