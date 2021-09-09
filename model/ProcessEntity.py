from enum import Enum

from synthetic_data_generator.gui import Colors


class ProcessState(str, Enum):
    """
    A ProcessEntity can have the following states:
    RUNNING: The process has not terminated yet.
    SUCCESSFUL: The process has regularly terminated without any error.
    FAILED: The process has irregularly terminated due to an error.
    ABORTED: The process was killed before terminating.
    """
    RUNNING = "running"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    ABORTED = "aborted"

    def to_color(self) -> tuple:
        """
        :return: a color value to visualize the ProcessState.
        """
        if self == self.RUNNING:
            return Colors.yellow
        elif self == self.SUCCESSFUL:
            return Colors.green
        else:
            return Colors.red


class ProcessEntityMixin:
    """
    A ProcessEntityMixin holds the state of an entity representing a process.

    It is a Mixin, so it should not be instantiated. As it has no abstract methods, it is no true abstract class and
    hence only marked as Mixin. Objects that are used to start processes with the ProcessManager should inherit from
    this class to obtain common behaviour regarding state changes.
    """

    def __init__(self, state: ProcessState):
        """
        Initialize the entity with its initial :param state:.
        """
        self.state = state

    def finished_successfully(self):
        """
        Indicate a state change to State.SUCCESSFUL
        """
        self.state = ProcessState.SUCCESSFUL

    def failed(self):
        """
        Indicate a state change to State.FAILED
        """
        self.state = ProcessState.FAILED

    def aborted(self):
        """
        Indicate a state change to State.ABORTED
        """
        self.state = ProcessState.ABORTED
