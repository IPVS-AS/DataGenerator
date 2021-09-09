from synthetic_data_generator.model.FileEntity import FileEntity


class ProgramConfig(FileEntity):
    """
    Models all parameters that used for the general program configuration.
    """

    def __init__(self,
                 run_number: int = 1,
                 mlc_number: int = 1):
        """
        Create a config for the general configuration of the ImbalanceDataGenerator:
        :param run_number: number of runs used for automatic naming
        :param mlc_number: number of MachineLearningConfigs used for automatic name suggestions
        """
        FileEntity.__init__(self, 'program_config')
        self.run_number = run_number
        self.mlc_number = mlc_number
