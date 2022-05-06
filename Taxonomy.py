from anytree import NodeMixin, RenderTree


class NodeInformation:
    def __init__(self, feature_set, n_instances, data=None, n_classes=None, target=None, classes=None,
                 class_occurences=None):
        # these are required for the hierarchy specification
        self.feature_set = feature_set
        self.n_instances = n_instances
        self.n_classes = n_classes

        # here we define the overall data
        self.data = data
        self.target = target

        # additional information that can be specified, e.g., for a hard-coded hierarchy
        self.classes = classes
        self.class_occurences = class_occurences


class Node(NodeInformation, NodeMixin):
    """
    Node class. Keeps track of the information of one node in the whole hierarchy.
    Basically, the node needs to keep track of the number of instances (n_instances), features (n_features),
    classes (n_classes), and the parents/child nodes.
    This class is based on the NodeInformation/NodeMixin classes from anytree and extends them to also include the
    specific information.
    """

    def __init__(self, node_id, parent=None,n_instances=None, childrens=None, n_classes=None, data=None,
                 target=None,feature_set=None, classes=None,
                 class_occurences=None):
        super(Node, self).__init__(feature_set=feature_set, n_instances=n_instances, n_classes=n_classes, data=data,
                                   target=target, classes=classes,
                                   class_occurences=class_occurences)
        self.feature_set = feature_set
        if self.children:
            self.children.extend(childrens)
        else:
            self.children = []
        self.parent = parent

        if self.parent:
            # setting format for level and the values for each level. We start with level 0 until max depth
            # From left to right the different values are the node ids (from 0 to x)
            self.level = self.parent.level + 1
            self.hierarchy_level_values = self.parent.hierarchy_level_values.copy()
            # So we only keep track of the node ids of the parents -> makes it easier to access them later on
            self.hierarchy_level_values[self.parent.level] = self.parent.node_id
        else:
            self.level = 0
            self.hierarchy_level_values = {}
        self.name = node_id
        self.node_id = node_id
        self.target = target

        # additional information that would be nice if the class occurences are known
        self.gini = None

    def has_child_nodes(self):
        return len(self.children) > 0

    def get_child_nodes(self):
        return self.children

    def append_child(self, child):
        self.children.append(child)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.classes:
            return f"Level-{self.level};{self.node_id}[n_instances={self.n_instances}," \
                   f" n_classes={self.n_classes}]"

        return f"Level-{self.level};{self.node_id}"

    def remove_children(self):
        self.children = list()


class EngineTaxonomy:
    """
    Represents a 'hardcoded' hierarchy that is very close to the Hierarchy from the hirsch et al. paper.
    We define exactly how many samples, classes, features and even how often each class occurs.
    """

    def __init__(self):
        pass

    def create_taxonomy(self):
        # level 0
        root = Node(node_id="Engine")

        # level 1
        DE = Node(node_id="Diesel", parent=root)
        # ## all classes for level 2 and 3 that belong to DE have to have the classes in the same range as DE, i.e.,
        # (1, 60)

        # level 2
        om1 = Node(node_id="DE-OM1", parent=DE)

        # level 3
        # n_instances=1 are removed --> we add them to another class
        # om1_1 = Node(node_id="DE-OM1-1", n_instances=1, parent=om1, feature_set=None)
        om1_2 = Node(node_id="DE-OM1-2", parent=om1)
        om1_3 = Node(node_id="DE-OM1-3", parent=om1)
        om1_4 = Node(node_id="DE-OM1-4", parent=om1)
        #om1_5 = Node(node_id="DE-OM1-5", parent=om1)
        #om1_6 = Node(node_id="DE-OM1-6", parent=om1)

        # level 2
        om2 = Node(node_id="DE-OM2", parent=DE)

        # level 3
        om2_1 = Node(node_id="DE-OM2-1", parent=om2)
        om2_2 = Node(node_id="DE-OM2-2", parent=om2)
        om2_3 = Node(node_id="DE-OM2-3", parent=om2)
        # om2_4 = Node(node_id="DE-OM2-4", n_instances=1, parent=om2, feature_set=None)
        om2_5 = Node(node_id="DE-OM2-5", parent=om2)
        om2_6 = Node(node_id="DE-OM2-6", parent=om2)

        # level 2
        om3 = Node(node_id="DE-OM3",parent=DE)
        # level 3
        om3_1 = Node(node_id="DE-OM3-1", parent=om3)
        # om3_2 = Node(node_id="DE-OM3-2", n_instances=1, parent=om3, feature_set=None)
        om3_3 = Node(node_id="DE-OM3-3", parent=om3)

        # level 1
        GE = Node(node_id="Gasoline", parent=root)

        # level 2
        # Info: Freq(A) = 51, b: 13, C: 13, D:9
        # --> The rest are 39 classes that occur around 2 or 3 times
        GE_om1 = Node(node_id="GE-OM1", parent=GE)
        # level 3
        # GE_om1_1 = Node(node_id="GE-OM1-1", n_instances=1, parent=GE_om1, feature_set=None)
        GE_om1_2 = Node(node_id="GE-OM1-2", parent=GE_om1)
        GE_om1_3 = Node(node_id="GE-OM1-3", parent=GE_om1)
        GE_om1_4 = Node(node_id="GE-OM1-4", parent=GE_om1)

        GE_om1_5 = Node(node_id="GE-OM1-5", parent=GE_om1)
        GE_om1_6 = Node(node_id="GE-OM1-6", parent=GE_om1)
        GE_om1_7 = Node(node_id="GE-OM1-7", parent=GE_om1)

        # level 2
        GE_om3 = Node(node_id="GE-OM3", parent=GE)
        # level 3
        GE_om3_1 = Node(node_id="GE-OM3-1", parent=GE_om3)
        # GE_om3_2 = Node(node_id="GE-OM3-2", n_instances=1, parent=GE_om3, feature_set=None)
        # GE_om3_3 = Node(node_id="GE-OM3-3", n_instances=1, parent=GE_om3, feature_set=None)
        GE_om3_4 = Node(node_id="GE-OM3-4", parent=GE_om3)
        #GE_om3_5 = Node(node_id="GE-OM3-5", parent=GE_om3)
        GE_om3_6 = Node(node_id="GE-OM3-6", parent=GE_om3)
        GE_om3_7 = Node(node_id="GE-OM3-7", parent=GE_om3)
        GE_om3_8 = Node(node_id="GE-OM3-8", parent=GE_om3)
        GE_om3_9 = Node(node_id="GE-OM3-9",parent=GE_om3)
        GE_om3_10 = Node(node_id="GE-OM3-10",parent=GE_om3)
        GE_om3_11 = Node(node_id="GE-OM3-11", parent=GE_om3)
        GE_om3_12 = Node(node_id="GE-OM3-12", parent=GE_om3)
        #GE_om3_13 = Node(node_id="GE-OM3-13",parent=GE_om3)
        return root

    def name(self):
        return "engine-taxonomy"


if __name__ == '__main__':
    root = EngineTaxonomy().create_taxonomy()
    print(RenderTree(root))