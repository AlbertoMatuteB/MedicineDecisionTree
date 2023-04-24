import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

class Node:
    """
    Class representing a node in the decision tree.
    """
    def __init__(self, split_feature=None, split_value=None, label=None, left_child=None, right_child=None):
        self.split_feature = split_feature  # Index of the feature used for splitting at this node
        self.split_value = split_value  # Value of the feature used for splitting at this node
        self.label = label  # Label at this node (if it's a leaf)
        self.left_child = left_child  # Left child node
        self.right_child = right_child  # Right child node

class DecisionTree:
    """
    Class representing a decision tree classifier.
    """
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the decision tree
        self.tree = None  # Root node of the decision tree

    def fit(self, X, y):
        """
        Fit the decision tree to the given data.

        Parameters:
            -- X (numpy array): Features data, shape (n_samples, n_features).
            -- y (numpy array): Labels data, shape (n_samples,).

        Returns:
            -- None
        """
        self.tree = self._build_tree(X, y)  # Call the recursive _build_tree function to construct the tree

    def _build_tree(self, X, y):
        """
        Recursive function to build the decision tree.

        Parameters:
            -- X (numpy array): Features data, shape (n_samples, n_features).
            -- y (numpy array): Labels data, shape (n_samples,).

        Returns:
            -- Node: Root node of the decision tree.
        """
        # Base case: if all samples have the same label, return a leaf node with that label
        if np.all(y == y[0]):
            return Node(label=y[0])

        # Base case: if maximum depth is reached, return a leaf node with the majority label
        if self.max_depth is not None and self.max_depth == 0:
            majority_label = np.argmax(np.bincount(y))
            return Node(label=majority_label)

        # Find the best split using entropy
        best_split_feature, best_split_value = self._find_best_split(X, y)

        # Split the data based on the best split
        X_left, y_left, X_right, y_right = self._split_data(X, y, best_split_feature, best_split_value)

        # Recursive call to build left and right subtrees
        left_child = self._build_tree(X_left, y_left)
        right_child = self._build_tree(X_right, y_right)

        # Create a new node with the best split feature and value, and set its children
        node = Node(split_feature=best_split_feature, split_value=best_split_value,
                    left_child=left_child, right_child=right_child)

        return node

    def _find_best_split(self, X, y):
        """
        Find the best split for the decision tree using entropy as the splitting criterion.

        Parameters:
            -- X (numpy array): Features data, shape (n_samples, n_features).
            -- y (numpy array): Labels data, shape (n_samples,).

        Returns:
            -- int: Index of the best split feature.
            -- float: Value of the best split feature.
        """
        best_entropy = float('inf')
        best_split_feature= None
        best_split_value = None

        n_samples, n_features = X.shape

        # Loop through all features
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])  # Get unique values of the current feature

            # Loop through all unique values of the current feature
            for value in unique_values:
                # Split the data based on the current feature and value
                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, value)

                # Calculate the entropy for the left and right splits
                entropy_left = self._calculate_entropy(y_left)
                entropy_right = self._calculate_entropy(y_right)

                # Calculate the weighted average entropy for the current split
                weighted_entropy = (len(y_left) / n_samples) * entropy_left + (len(y_right) / n_samples) * entropy_right

                # Update the best split if the current weighted entropy is lower than the current best entropy
                if weighted_entropy < best_entropy:
                    best_entropy = weighted_entropy
                    best_split_feature = feature_idx
                    best_split_value = value

        return best_split_feature, best_split_value

    def _split_data(self, X, y, feature_idx, value):
        """
        Split the data based on a given feature and value.

        Parameters:
            -- X (numpy array): Features data, shape (n_samples, n_features).
            -- y (numpy array): Labels data, shape (n_samples,).
            -- feature_idx (int): Index of the feature used for splitting.
            -- value (float or str): Value of the feature used for splitting.

        Returns:
            -- numpy array: Left split of features data.
            -- numpy array: Left split of labels data.
            -- numpy array: Right split of features data.
            -- numpy array: Right split of labels data.
        """
        if isinstance(value, str):
            # Categorical feature
            mask = X[:, feature_idx] == value
        else:
            # Numerical feature
            mask = X[:, feature_idx] <= value

        X_left = X[mask]
        y_left = y[mask]
        X_right = X[~mask]
        y_right = y[~mask]

        return X_left, y_left, X_right, y_right

    def _calculate_entropy(self, y):
        """
        Calculate the entropy of the given labels data.

        Parameters:
            -- y (numpy array): Labels data, shape (n_samples,).

        Returns:
            -- float: Entropy.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-7))

        return entropy

    def predict(self, X):
        """
        Predict the labels for the given data.

        Parameters:
            -- X (numpy array): Features data, shape (n_samples, n_features).

        Returns:
            -- numpy array: Predicted labels, shape (n_samples,).
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the decision tree to predict the label for a single sample.

        Parameters:
            -- x (numpy array): Single sample features data, shape (n_features,).
            -- node (Node): Current node of the decision tree.

        Returns:
            -- int: Predicted label for the sample.
        """
        if node.label is not None:
            return node.label

        if isinstance(x[node.split_feature], str):
            # Categorical feature
            if x[node.split_feature] == node.split_value:
                return self._traverse_tree(x, node.left_child)
            else:
                return self._traverse_tree(x, node.right_child)
        else:
            # Numerical feature
            if x[node.split_feature] <= node.split_value:
                return self._traverse_tree(x, node.left_child)
            else:
                return self._traverse_tree(x, node.right_child)
            
    def get_tree(self):
        """
        Get the trained decision tree.

        Returns:
            -- Node: Root node of the decision tree.
        """
        return self.tree

    def _print_tree(self, node, depth=0):
        """
        Print the decision tree for visualization purposes.

        Parameters:
            -- node (Node): Current node of the decision tree.
            -- depth (int): Depth of the current node in the tree.
        """
        if node.label is not None:
            print('  ' * depth + f'Predicted class: {node.label}')
        else:
            print('  ' * depth + f'{node.split_feature} <= {node.split_value}')
            self._print_tree(node.left_child, depth + 1)
            print('  ' * depth + f'{node.split_feature} > {node.split_value}')
            self._print_tree(node.right_child, depth + 1)

    def print_tree(self):
        """
        Print the decision tree for visualization purposes.
        """
        self._print_tree(self.tree)

# load sex input in numeric form
def input_sex():
    print("Enter sex (M or F): ")
    sex = input()
    if sex == 'M' or sex.lower() == 'male':
        return 0
    elif sex == 'F' or sex.lower() == 'female':
        return 1
    else:
        print("Invalid input. Please try again.")

# load blood pressure input in numeric form
def input_bp():
    print("Enter blood pressure (LOW, NORMAL, or HIGH): ")
    bp = input()
    if bp.lower() == 'low':
        return 0
    elif bp.lower() == 'normal':
        return 1
    elif bp.lower() == 'high':
        return 2
    else:
        print("Invalid input. Please try again.")

# load cholesterol input in numeric form
def input_chol():
    print("Enter cholesterol (NORMAL or HIGH): ")
    chol = input()
    if chol.lower() == 'normal':
        return 0
    elif chol.lower() == 'high':
        return 1
    else:
        print("Invalid input. Please try again.")

# Load the dataset from CSV using pandas
df = pd.read_csv('DecisionTreeProject1\classification_1\drug200.csv')

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Transform the categorical values into numerical values using LabelEncoder

# For 'sex' column
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# For 'BP' column
df['BP'] = label_encoder.fit_transform(df['BP'])

# For 'cholesterol' column
df['Cholesterol'] = label_encoder.fit_transform(df['Cholesterol'])


# Extract features (X) and labels (y) from the DataFrame
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df['Drug'].values

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTree(max_depth=3)
clf.fit(X_train, y_train)


# Print the trained tree
clf.print_tree()

# predict using the test data
y_pred = clf.predict(X_test)


# do the same using sklearn
from sklearn.tree import DecisionTreeClassifier

# Train a decision tree classifier
clf_fw = DecisionTreeClassifier(criterion="entropy", max_depth = 3)
clf_fw.fit(X_train, y_train)

# Print the trained tree

tree.plot_tree(clf_fw)
plt.show()

# Predict labels for clf and clf_fw
y_pred = clf.predict(X_test)
y_pred_fw = clf_fw.predict(X_test)

print("My prediction:")
print (y_pred [0:5])
print (y_test [0:5])

print("Sklearn prediction:")
print (y_pred_fw [0:5])
print (y_test [0:5])

#print accuracy score for both
print("My accuracy score:")
print (np.mean(y_pred == y_test))

print("Sklearn accuracy score:")
print (np.mean(y_pred_fw == y_test))

# predict using user input
# Create variables based on user input
age = int(input("Enter age: "))
sex = input_sex()
bp = input_bp()
chol = input_chol()
na_to_k = float(input("Enter Na to K ratio: "))

# Create a numpy array based on user input
user_input = [age,sex,bp,chol,na_to_k]

# Predict the label using the trained tree
prediction = clf.predict(np.array(user_input).reshape(1, -1))

# Print the predicted label
print(f'Predicted label: {prediction[0]}')
