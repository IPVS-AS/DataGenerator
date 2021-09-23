from abc import ABC


class FileEntity(ABC):
    """
    This class encapsulates common data and functionality for domain objects that are persisted as entities files.
    """

    def __init__(self, name: str, writeable: bool = True):
        """
        Entities persisted in the file system will be identified based on their name only.
        Therefore, this class manifests the unique name assumption by providing the self.name attribute for all objects
        inheriting from this class.
        :param name: the self.name attribute to be stored as a uniquely identifying name
        :param writeable: Additionally, it provides the attribute self._writeable to note if an entity is editable, or
        read-only. Per default, an entity is writeable, so the attribute defaults to True.
        """
        self.name = name
        self._writeable = writeable

    def equals(self, entity_to_compare) -> bool:
        """
        This operation checks whether an entity is equal based on its type and the unique name assumption.
        :param entity_to_compare: the object must have an attribute entity_to_compare.name
        :return: True, iff self.name == entity_to_compare.name AND type(self) == type(entity_to_compare)
        """
        return (self.name == entity_to_compare.name) and (type(self) == type(entity_to_compare))

    def _make_read_only(self):
        """
        To be used by `~repository.FileRepository.FileRepository` only.
        Marks the entity as non-writeable.
        """
        self._writeable = False

    def is_writeable(self) -> bool:
        """
        :return: True, if the entity is still writeable || False, if the entity is read-only
        """
        return self._writeable
