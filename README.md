## Decision Tree Classifier
This code implements a decision tree classifier to perform binary classification tasks. It consists of two classes: Node and DecisionTree.

# Node class
The Node class represents a node in the decision tree. Each node can be a split node, which contains information about the split feature and value, or a leaf node, which contains the predicted class label. The class is defined as follows:

```
class Node:
    def __init__(self, split_feature=None, split_value=None, label=None, left_child=None, right_child=None):
        self.split_feature = split_feature  # Index of the feature used for splitting at this node
        self.split_value = split_value  # Value of the feature used for splitting at this node
        self.label = label  # Label at this node (if it's a leaf)
        self.left_child = left_child  # Left child node
        self.right_child = right_child  # Right child node
```

# DecisionTree class
The DecisionTree class represents the decision tree classifier. It is defined as follows:

```
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the decision tree
        self.tree = None  # Root node of the decision tree
```
The class has two methods: `fit` and `_build_tree`. The fit method is used to fit the decision tree to the given data. The _build_tree method is a recursive function that is used to build the decision tree.

The other functions in the DecisionTree class are helper functions used in the implementation of the decision tree algorithm.

`_calculate_entropy(y)`: Calculates the entropy of a set of labels y. The entropy is a measure of the impurity of a set of labels, and is defined as - sum(p * log2(p)) where p is the proportion of samples in the set that belong to each class. The _calculate_entropy function takes as input the labels y and returns the entropy as a float.

`_split_data(X, y, feature_idx, value)`: Splits the features data X and labels data y based on a given feature feature_idx and value value. This function returns four numpy arrays: the left split of the features data, the left split of the labels data, the right split of the features data, and the right split of the labels data.

`predict(X)`: Predicts the labels of a set of samples X using the decision tree. This function takes as input the features data X, and returns a numpy array of predicted labels for each sample in X. The _predict_one function is called on each sample in X using the root node of the tree as the starting point for the recursion.
