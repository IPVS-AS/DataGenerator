from pandas import DataFrame

from synthetic_data_generator.model.FileEntity import FileEntity


class DataGenerationResult(FileEntity):
    """
    The Result of a Data Generation Run (dgr), identified with the same name as the corresponding dgr.
    """

    def __init__(self, name: str, data_frame: DataFrame):
        """
        Wrap the pandas data frame in a domain entity by additionally specifying its name.
        DataGenerationResults are only written once, therefore no parameter writeable is specified here.

        :param name: the uniquely identifying name of the dgr to match the result to the corresponding run
        :param data_frame: the pandas.DataFrame containing the result of the data generator
        """
        super().__init__(name, True)
        self.data_frame = data_frame
