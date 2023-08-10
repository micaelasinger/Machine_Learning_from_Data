import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    _, counts = np.unique(data[:, -1], return_counts=True)
    p = counts / data.shape[0]
    gini = 1 - np.sum(p ** 2)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    la_uni = (np.unique(data[:, -1], return_counts=True)[1] / data.shape[0])
    entropy = np.dot(la_uni, np.log2(la_uni))
    return -entropy


def calc_gain_ratio(data, feature, goodness):
    samples = data.shape[0]
    _, counts = np.unique(data[:, feature], return_counts=True)
    distribution = counts / samples
    entropy = - np.sum(distribution * np.log2(distribution))
    return goodness / entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    samples_amount = data.shape[0]
    parent_impurity = impurity_func(data)
    unique_values = np.unique(data[:, feature], return_counts=True)
    data_subsets_by_feature_value = [data[data[:, feature] == val] for val in unique_values[0]]

    distributions = unique_values[1] / samples_amount
    groups = dict(zip(unique_values[0], data_subsets_by_feature_value))
    impurity_results = list(map(impurity_func, data_subsets_by_feature_value))

    goodness = parent_impurity - np.sum(np.multiply(distributions, impurity_results))

    if gain_ratio:
        goodness = calc_gain_ratio(data, feature, goodness)

    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.labels, self.counts = np.unique(self.data[:, -1], return_counts=True)
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio
        self.num_features = self.data.shape[1] - 1
        self.goodness = -1
        self.chi_sqr = None

    def chi_sqr_update(self):
        """
        Calculate the chi square value of the node.
        This method assumes there are two labels!

        Returns:
        - chi_sqr: the chi square value of the node.
        """
        chi_sqr = 0
        py0 = self.counts[0] / self.data.shape[0]
        py1 = self.counts[1] / self.data.shape[0]
        vals = np.unique(self.data[:, self.feature]).tolist()
        for i in vals:
            df = (self.data[:, self.feature] == i).sum()
            pf = ((self.data[:, self.feature] == i) & (self.data[:, -1] == self.labels[0])).sum()
            nf = ((self.data[:, self.feature] == i) & (self.data[:, -1] == self.labels[1])).sum()
            e0 = df * py0
            e1 = df * py1
            chi_sqr = chi_sqr + (np.square(pf - e0) / e0) + (np.square(nf - e1) / e1)

        self.chi_sqr = chi_sqr

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        if self.data.shape[0] > 0 and self.data.shape[1] > 0:
            pred = self.labels[np.argmax(self.counts)]
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        if len(self.labels) == 1:
            self.terminal = True
            return
        if self.depth >= self.max_depth:
            self.terminal = True
            return
        if self.num_features == 0:
            print(f'No features: depth {self.depth} data_size is {self.data.shape}')
            self.terminal = True
            return

        best_split_values = None
        for f in range(self.num_features):
            if np.unique(self.data[:, f]).shape[0] <= 1:
                continue
            goodness, split_values = goodness_of_split(self.data, f, impurity_func, gain_ratio=self.gain_ratio)
            if goodness > self.goodness:
                self.goodness = goodness
                best_split_values = split_values
                self.feature = f

        if len(best_split_values) > 1:
            self.chi_sqr_update()

            df = len(best_split_values) - 1
            if (self.chi == 1) or (self.chi_sqr >= chi_table[df][self.chi]):
                for val, sub_data in best_split_values.items():
                    child_node = DecisionNode(sub_data, feature=-1, depth=self.depth + 1, chi=self.chi,
                                              max_depth=self.max_depth, gain_ratio=self.gain_ratio)
                    self.add_child(child_node, val)
            else:
                self.terminal = True
        else:
            print(f'No more possible splits: num_features {self.num_features} data_size is {self.data.shape}')
            self.terminal = True


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    queue = [root]
    while len(queue) > 0:
        node = queue.pop(0)
        node.split(impurity)
        if not node.terminal:
            queue += node.children
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    while not root.terminal:
        f = root.feature
        try:
            idx = root.children_values.index(instance[f])
        except Exception as e:
            break
        root = root.children[idx]

    pred = root.pred
    return pred

def predict_aux(instance, root):
    return predict(root, instance)

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    pred = np.apply_along_axis(predict_aux, axis=1, arr=dataset, root=node)
    res = dataset[:, -1] == pred
    accuracy = res.sum() / res.shape[0]
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree_entropy_gain_ratio_training = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True,
                                                      max_depth=max_depth)
        training_accuracy = calc_accuracy(tree_entropy_gain_ratio_training, X_train)
        testing_accuracy = calc_accuracy(tree_entropy_gain_ratio_training, X_test)
        training.append(training_accuracy)
        testing.append(testing_accuracy)
    return training, testing

def calc_tree_depth(root):
    root_arr = np.array([root])
    depth = 0
    while root_arr.size:
        depth += 1
        new_root_arr = np.array([])
        for n in root_arr:
            new_root_arr = np.concatenate([new_root_arr, n.children])
        root_arr = new_root_arr
    return depth

def chi_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []
    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, max_depth=1000,
                                             chi=chi)  # entropy and gain ratio
        train_acc = calc_accuracy(tree_entropy_gain_ratio, X_train)
        test_acc = calc_accuracy(tree_entropy_gain_ratio, X_test)
        depth.append(calc_tree_depth(tree_entropy_gain_ratio))
        chi_training_acc.append(train_acc)
        chi_testing_acc.append(test_acc)
    return chi_training_acc, chi_testing_acc, depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = 1
    for i in node.children:
        n_nodes += count_nodes(i)
    return n_nodes






