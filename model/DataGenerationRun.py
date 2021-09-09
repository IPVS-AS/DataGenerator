"""
This module provides a class for managing DataGenerationRun-entities as well as corresponding constants as enumerations.
"""

from enum import Enum
from synthetic_data_generator.model.FileEntity import FileEntity
from synthetic_data_generator.model.ProcessEntity import ProcessState, ProcessEntityMixin


class ImbalanceDegree(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DataGenerationRun(FileEntity, ProcessEntityMixin):
    """
    Models all parameters that make up a run of the Imbalance Generator.
    """

    def __init__(self,
                 name: str,
                 hierarchy_used: str,
                 imbalance_degree: ImbalanceDegree,
                 seed: int,
                 feature_remove_percent: float,
                 n_features: int,
                 noise: float,
                 writeable: bool = True,
                 state: ProcessState = ProcessState.RUNNING):
        """
        Create a new data generation run with the following attributes:
        :param name: the unique name of the entity
        :param hierarchy_used: the name of the hierarchy to be used by the imbalance generator
        :param imbalance_degree: the imbalance degree of the data generation
        :param seed: the seed of the data generation
        :param feature_remove_percent: the percentage of features to remove in data generation
        :param n_features: the count of all features at the beginning
        :param noise: the noise parameter of  the algorithm
        :param writeable: [OPTIONAL] default: True. Marks the entity as read-only if set to False.
        :param state: [OPTIONAL] the state, will be State.RUNNING initially if parameter is not used
        """
        FileEntity.__init__(self, name, writeable)
        ProcessEntityMixin.__init__(self, state)
        self.hierarchy_used = hierarchy_used
        self.imbalance_degree = imbalance_degree
        self.seed = seed
        self.feature_remove_percent = feature_remove_percent
        self.n_features = n_features
        self.noise = noise
