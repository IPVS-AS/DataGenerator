from anytree import RenderTree, PreOrderIter
from anytree import NodeMixin, RenderTree
import numpy as np

np.random.seed(1234)


class NodeInformation:
    def __init__(self, feature_set, n_samples, data=None, n_classes=None, target=None, training_data=None,
                 training_labels=None, classes=None, class_occurences=None):
        # these are required for the hierarchy specification
        self.feature_set = feature_set
        self.n_samples = n_samples
        self.n_classes = n_classes

        # here we define the overall data
        self.data = data
        self.target = target

        # define what is training data --> makes it easier to store them in the nodes for running the machine learning
        # algorithms
        self.training_data = training_data
        self.training_labels = training_labels

        # custom data after each of the machine learning steps
        self.sph_data = None
        self.sph_labels = None
        self.cpi_data = None
        self.cpi_labels = None

        # additional information that can be specified, e.g., for a hard-coded hierarchy
        self.classes = classes
        self.class_counter = None
        self.class_occurences = class_occurences


class Node(NodeInformation, NodeMixin):
    """
    Node class. Keeps track of the information of one node in the whole hierarchy.
    Basically, the node needs to keep track of the number of sampples (n_samples), features (n_features),
    classes (n_classes), and the parents/child nodes.
    This class is based on the NodeInformation/NodeMixin classes from anytree and extends them to also include the
    specific information.
    """

    def __init__(self, feature_set, n_samples, node_id, parent=None, childrens=None, n_classes=None, data=None,
                 target=None, training_data=None, classes=None, class_occurences=None):
        super(Node, self).__init__(feature_set=feature_set, n_samples=n_samples, n_classes=n_classes, data=data,
                                   target=target, training_data=training_data, classes=classes,
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
        if self.classes and self.gini:
            # nice to see gini as well if we set it
            return f"Level-{self.level};{self.node_id}[n_samples={self.n_samples}," \
                   f" n_classes={self.n_classes}, " \
                   f"gini={round(self.gini, 2)}]]"
        else:
            return f"Level-{self.level};{self.node_id}[n_samples={self.n_samples}," \
                   f" n_classes={self.n_classes}]"


class HardCodedHierarchy:
    """
    Represents a 'hardcoded' hierarchy that is very close to the Hierarchy from the hirsch et al. paper.
    We define exactly how many samples, classes, features and even how often each class occurs.
    """

    def __init__(self):
        pass

    def create_hardcoded_hierarchy(self):
        # level 0
        root = Node(node_id="Engine", n_samples=1050, feature_set=None, n_classes=84, classes=(1, 84))

        # level 1
        hde = Node(node_id="hde", n_samples=277, parent=root, feature_set=None, n_classes=60, classes=(1, 60))
        # ## all classes for level 2 and 3 that belong to hde have to have the classes in the same range as hde, i.e.,
        # (1, 60)

        # level 2
        om1 = Node(node_id="hde-OM1", n_samples=96, parent=hde, feature_set=None, n_classes=28, classes=(1, 28))

        # level 3
        # n_samples=1 are removed --> we add them to another class
        # om1_1 = Node(node_id="hde-OM1-1", n_samples=1, parent=om1, feature_set=None)
        om1_2 = Node(node_id="hde-OM1-2", n_samples=10, parent=om1, feature_set=None, n_classes=4, classes=(1, 4),
                     class_occurences=[2, 3, 2, 3])
        om1_3 = Node(node_id="hde-OM1-3", n_samples=37, parent=om1, feature_set=None, n_classes=10, classes=(5, 14),
                     class_occurences=[8, 5, 3, 3, 3, 3] + [3 for _ in range(11, 15)])
        om1_4 = Node(node_id="hde-OM1-4", n_samples=15, parent=om1, feature_set=None, n_classes=5, classes=(15, 19),
                     class_occurences=[6, 4, 3, 1, 1])
        om1_5 = Node(node_id="hde-OM1-5", n_samples=22, parent=om1, feature_set=None, n_classes=5, classes=(20, 24),
                     class_occurences=[10, 5, 5, 1, 1])
        om1_6 = Node(node_id="hde-OM1-6", n_samples=12, parent=om1, feature_set=None, n_classes=4, classes=(25, 28),
                     class_occurences=[1, 1, 7, 3])
        # om1_7= Node(node_id="hde-OM1-7", n_samples=4, parent=om1, feature_set=None, n_classes=2, classes=(29, 30),
        #             class_occurences=[3, 1])

        # level 2
        om2 = Node(node_id="hde-OM2", n_samples=130, parent=hde, feature_set=None, n_classes=41, classes=(5, 45))

        # level 3
        om2_1 = Node(node_id="hde-OM2-1", n_samples=52, parent=om2, feature_set=None, n_classes=20, classes=(5, 24),
                     class_occurences=[6, 5, 4, 3, 3, 3, 4, 3, 3, 4, 3, 3] + [1 for _ in range(17, 25)])
        om2_2 = Node(node_id="hde-OM2-2", n_samples=12, parent=om2, feature_set=None, n_classes=9, classes=(25, 33),
                     class_occurences=[1 for _ in range(27, 32)] + [3, 2, 1, 1])
        om2_3 = Node(node_id="hde-OM2-3", n_samples=8, parent=om2, feature_set=None, n_classes=2, classes=(33, 34),
                     class_occurences=[5, 3])
        # om2_4 = Node(node_id="hde-OM2-4", n_samples=1, parent=om2, feature_set=None)
        om2_5 = Node(node_id="hde-OM2-5", n_samples=43, parent=om2, feature_set=None, n_classes=13, classes=(32, 44),
                     class_occurences=[8, 4, 5, 4, 3, 3, 2, 1] + [2 for _ in range(38, 42)] + [5])
        om2_6 = Node(node_id="hde-OM2-6", n_samples=15, parent=om2, feature_set=None, n_classes=4, classes=(45, 48),
                     class_occurences=[10, 2, 1, 2])

        # level 2
        om3 = Node(node_id="hde-OM3", n_samples=51, parent=hde, feature_set=None, n_classes=12, classes=(41, 52))
        # level 3
        om3_1 = Node(node_id="hde-OM3-1", n_samples=8 + 1, parent=om3, feature_set=None, n_classes=3, classes=(41, 43),
                     class_occurences=[7, 1, 1])
        # om3_2 = Node(node_id="hde-OM3-2", n_samples=1, parent=om3, feature_set=None)
        om3_3 = Node(node_id="hde-OM3-3", n_samples=42, parent=om3, feature_set=None, n_classes=11, classes=(42, 52),
                     class_occurences=[2, 7] + [14, 1, 2, 3, 5] + [2 for _ in range(51, 55)])

        # level 1
        mde = Node(node_id="mde", n_samples=773, parent=root, feature_set=None, n_classes=58, classes=(27, 84))

        # level 2
        # Info: Freq(A) = 51, b: 13, C: 13, D:9
        # --> The rest are 39 classes that occur around 2 or 3 times
        mde_om1 = Node(node_id="mde-OM1", n_samples=200, parent=mde, feature_set=None, n_classes=43, classes=(27, 69))
        # level 3
        # mde_om1_1 = Node(node_id="mde-OM1-1", n_samples=1, parent=mde_om1, feature_set=None)
        mde_om1_2 = Node(node_id="mde-OM1-2", n_samples=25, parent=mde_om1, feature_set=None, n_classes=8,
                         classes=(27, 34), class_occurences=[1, 1, 1, 1, 4, 4, 8, 5])
        mde_om1_3 = Node(node_id="mde-OM1-3", n_samples=81, parent=mde_om1, feature_set=None, n_classes=20,
                         classes=(35, 54),
                         class_occurences=[3 for _ in range(35, 42)] + [2 for _ in range(42, 51)] + [3, 33]+ [3 for _
                                                                                                               in
                                                                                                               range(55,
                                                                                                                     57)])
        mde_om1_4 = Node(node_id="mde-OM1-4", n_samples=65, parent=mde_om1, feature_set=None, n_classes=15,
                         classes=(54, 68), class_occurences=[17, 13, 11, 7, 5, 2, 2] + [1 for _ in range(61, 69)])

        mde_om1_5 = Node(node_id="mde-OM1-5", n_samples=3 + 1, parent=mde_om1, feature_set=None, n_classes=3,
                         classes=(60, 62), class_occurences=[1, 2, 1])
        mde_om1_6 = Node(node_id="mde-OM1-6", n_samples=17, parent=mde_om1, feature_set=None, n_classes=7,
                         classes=(62, 68),
                         class_occurences=[5, 2, 2, 2] + [2 for _ in range(66, 69)])
        mde_om1_7 = Node(node_id="mde-OM1-7", n_samples=8, parent=mde_om1, feature_set=None, n_classes=3,
                         classes=(67, 69),
                         class_occurences=[1, 2, 5])

        # level 2
        mde_om3 = Node(node_id="mde-OM3", n_samples=573, parent=mde, feature_set=None, n_classes=54, classes=(31, 84))
        # level 3
        mde_om3_1 = Node(node_id="mde-OM3-1", n_samples=7 + 1, parent=mde_om3, feature_set=None, n_classes=2,
                         classes=(31, 32), class_occurences=[7, 1])
        # mde_om3_2 = Node(node_id="mde-OM3-2", n_samples=1, parent=mde_om3, feature_set=None)
        # mde_om3_3 = Node(node_id="mde-OM3-3", n_samples=1, parent=mde_om3, feature_set=None)
        mde_om3_4 = Node(node_id="mde-OM3-4", n_samples=61, parent=mde_om3, feature_set=None, n_classes=21,
                         classes=(31, 51),
                         class_occurences=[10, 7, 4, 3, 3, 3] + [3 for _ in range(37, 45)] + [1 for _ in range(45, 52)])
        mde_om3_5 = Node(node_id="mde-OM3-5", n_samples=33, parent=mde_om3, feature_set=None, n_classes=10,
                         classes=(48, 57), class_occurences=[18, 1, 4, 4] + [1 for _ in range(52, 58)])
        mde_om3_6 = Node(node_id="mde-OM3-6", n_samples=35, parent=mde_om3, feature_set=None, n_classes=10,
                         classes=(50, 59), class_occurences=[16, 5, 3, 3, 3] + [1 for _ in range(55, 60)])
        mde_om3_7 = Node(node_id="mde-OM3-7", n_samples=22, parent=mde_om3, feature_set=None, n_classes=5,
                         classes=(60, 64),
                         class_occurences=[9, 6, 3, 2, 2])
        mde_om3_8 = Node(node_id="mde-OM3-8", n_samples=16, parent=mde_om3, feature_set=None, n_classes=3,
                         classes=(65, 67),
                         class_occurences=[10, 5, 1])
        mde_om3_9 = Node(node_id="mde-OM3-9", n_samples=42, parent=mde_om3, feature_set=None, n_classes=10,
                         classes=(63, 72), class_occurences=[20, 10, 3, 3] + [1 for _ in range(67, 73)])
        mde_om3_10 = Node(node_id="mde-OM3-10", n_samples=12, parent=mde_om3, feature_set=None, n_classes=9,
                          classes=(65, 73), class_occurences=[4] + [1 for _ in range(66, 74)])
        mde_om3_11 = Node(node_id="mde-OM3-11", n_samples=7 + 1, parent=mde_om3, feature_set=None, n_classes=2,
                          classes=(71, 72), class_occurences=[7, 1])
        mde_om3_12 = Node(node_id="mde-OM3-12", n_samples=27, parent=mde_om3, feature_set=None, n_classes=7,
                          classes=(72, 78), class_occurences=[10, 9, 1, 1, 2] + [2 for _ in range(77, 79)])
        ### This is an important case!! ####
        mde_om3_13 = Node(node_id="mde-OM3-13", n_samples=309, parent=mde_om3, feature_set=None, n_classes=40,
                          classes=(45, 84),
                          class_occurences=[95, 48, 45, 35, 1] + [1 for _ in range(50, 60)] + [33]
                                           + [1, 1, 1, 1]
                                           + [1, 1, 1, 1, 1]
                                           + [1, 1, 1, 1, 1]
                                           + [1, 1, 1, 1] + [4 for _ in range(79, 85)])
        return root


root = HardCodedHierarchy().create_hardcoded_hierarchy()
print(RenderTree(root))
nodes = [root]

# test that classes and n_classes is the same in each node
# this is only needed to test that the Hierarchy is not specified with some typos
while True:
    next_node = nodes.pop()
    print(next_node)
    assert len(range(next_node.classes[0], next_node.classes[1] + 1)) == next_node.n_classes
    if next_node.children:
        nodes.extend(next_node.children)
    if len(nodes) == 0:
        break

product_group_nodes = [node for node in PreOrderIter(root) if not node.children]
for node in product_group_nodes:
    print(f"current node: {node.node_id}")
    print(f"sum is: {sum(node.class_occurences)}")
    assert sum(node.class_occurences) == node.n_samples
    print(f"len(node.class_occurences): {len(node.class_occurences)} == {node.n_classes}: node.n_classes")
    assert len(node.class_occurences) == node.n_classes
