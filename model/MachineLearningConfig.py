from synthetic_data_generator.model.FileEntity import FileEntity
from synthetic_data_generator.model.ProcessEntity import ProcessEntityMixin, ProcessState


class SequenceDefinition:
    """
    A class to encapsulate the definition of value sequences used for the configuration of machine learning.
    """

    def __init__(self, start: float, stop: float, step: float):
        """
        Initialize the definition of a sequence that can be efficiently persisted.

        :param start: The start value of the sequence, included as first value of the value list

        :param stop: The stop value of the sequence,
        included as last value of the value list iff there exist an n so that start + n*step = stop

        :param step: The step value defines the difference between two neighboring values in the sequence

        Preconditions:
        start > 0
        start <= stop
        step > 0
        """
        self.start = start
        self.stop = stop
        self.step = step

    def get_values(self) -> list:
        """
        Returns a list of float values, called "values".

        values[0] = start
        values[n+1] = values[n] + step
        max(values) <= stop
        """
        values = []
        current_value = self.start
        while current_value <= self.stop:
            values.append(current_value)
            current_value += self.step
        return values

    def __str__(self):
        """
        String definition for displaying ProcessManagerView.
        """
        return f"[{self.start}, {self.stop}] S: {self.step}"


class MachineLearningConfig(FileEntity, ProcessEntityMixin):
    """
    Model the configuration required to start the machine learning.
    """

    def __init__(self,
                 name: str,
                 dg_result_used: str,
                 gini_thresholds: SequenceDefinition,
                 p_quantile: SequenceDefinition,
                 max_info_loss: SequenceDefinition,
                 n_runs: int,
                 writeable: bool = True,
                 state: ProcessState = ProcessState.RUNNING):
        """
        Create a new Machine Learning Configuration with the following attributes:

        :param name: the unique name of the entity

        :param dg_result_used: the data generation result used for machine learning

        :param gini_thresholds: a SequenceDefinition for the percentage of the threshold for the gini index.

        :param p_quantile: a SequenceDefinition for the percentage of the thresholds for the p_quantile.

        :param max_info_loss: a SequenceDefinition for the percentage of information loss.

        :param n_runs: Number of runs to perform. The runs differ in different seed values.

        :param writeable: [OPTIONAL] default: True. Marks the entity as read-only if set to False.

        :param state: [OPTIONAL] the state, will be State.RUNNING initially if parameter is not used
        """
        FileEntity.__init__(self, name, writeable)
        ProcessEntityMixin.__init__(self, state)
        self.dg_result_used = dg_result_used
        self.gini_thresholds = gini_thresholds
        self.p_quantile = p_quantile
        self.max_info_loss = max_info_loss
        self.n_runs = n_runs
