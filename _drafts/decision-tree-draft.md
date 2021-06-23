---
title: "Notes on decision tree classifiers/regressors"
excerpt: "My notes on the algorithm and mathematics of decision tree classifiers and regressors."
date: 2021-07-05
last_modified_at: false
header:
  overlay_image: /assets/images/06_2021/slim.code.png
  overlay_filter: 0.5
---



# Decision tree

Decision trees are a non-parametric supervised learning method for classification and regression. The goal is to create a model that learns a hierarchy of if/else questions, using the data features to predict the value of the target variable. Decision trees can be used for **categorical** and **continuous** target variables. They are relatively simple to understand and can be easily visualized using a **decision tree**.

It is useful to first learn about the different parts of a decision tree. At the top is the **root** node, which is the starting point and represents the entire dataset. At each step, the dataset is **split** into two (or more) sub-nodes based on the particular value of one of its features. Which features to utilize and what values to split the data points on is the main challenge on constructing the model. There are numerous algorithms that can be used and one of the most popular is the **classification and regression tree (CART)** algorithm, which will be discussed in detail in the rest of these notes. After a split, the resulting sub-nodes can either be a **leaf (terminal)** node or a **decision** node. The leaf node represents an end to a tree and will not be subject to further splitting. Decision nodes represents one where further splitting will occur. You can actually think of the root node as just the first decision node of the tree.

<figure>
 	<img src="/assets/images/06_2021/DecisionTree.png">
	<figcaption><b>Figure 1.</b> Diagram of a simple decision tree.</figcaption>
</figure>
# Building a decision tree

1. Begin with the entire dataset at the root node.
2. Search over all possible splits and find the one that is most informative about the target; each split only concerns one feature. A split typically involves splitting the data base on whether its feature value is <= or > than a **threshold** value. The aim is to find the split that minimized the **impurity** of the sub-nodes. Thus, most decision tree algorithms are **greedy** algorithm that find the local optimum solution at each node, but this does not necessarily lead to a globally optimal decision tree.
3. Use the most informative split (minimum impurity) to divide the dataset into "left" and "right" sub-nodes.
4. Repeat steps 2 and 3 to recursively partition the dataset until each leaf only contains a single target class / regression value. A leaf node containing only data points with the same target value is a called **pure**. However, a leaf node can be impure for at least two reasons. Firstly, there may simply not be enough information in the features to separate  data points with different target values. Secondly, construction of the tree may be halted if some restriction is met to reduce overfitting (e.g. maximum depth was reached).

## Prediction

Once a decision tree is learned from training on the data, a prediction is made for new data points by looking at which leaf node they fall on based on their feature values. The prediction for the leaf node is simply the **majority target** for decision tree classifiers or the **mean value of targets** for decision tree regressors. Note that for decision tree regressors, the predicted value will never be outside of the values in the training dataset (i.e. it does not extrapolate).

An addition note for `scikit-learn` users using `DecisionTreeClassifier`. After a model is trained, the `predict` method is giving the majority target at the leaf node. The `predict_proba` method will output the proportion of each target class in the leaf node. 

## Overfitting

Building a tree until all leaves are pure usually causes the model to overfit onto the training dataset. To prevent overfitting, one can perform **pre-pruning** to stop building the tree early on or **post-pruning** to remove or collapse nodes with little information. Some examples of pre-pruning include limiting the max depth of the three, limiting the maximum number of leaves, and requiring a minimum number of data point to continue splitting a node.

# Measures of impurity

Evaluating the impurity of nodes is essential for building a decision tree. After all, the goal is to find the combination in feature space that would accurately predict the target value (minimize impurity). If the leaf node is highly impure, then the decision path that led to it would not give accurate predictions. It is important to separate the i**mpurity of a node** from the **quality of the split**. The former is calculated based on the target variables in a single node, while the latter is calculated using the impurity of sub-nodes (and sometimes the parent node) of a split. I will discuss how impurity is calculated here and the quality of the split in a later section.

Here I will define the impurity function as **\\(H(S_{m})\\)**, where **\\(S_{m}\\)** represents the set of data points at node \\(m\\). Impurity will obviously be calculated different for categorical vs continuous target variables. I will only present a few common measures here. 

## Classification measures of impurity

Given a categorical target variable with \(k\\) classes  (0, 1, ..., *k*-1, *k*). Let the proportion of each class at leaf node \\(m\\) be \\(p_{m, k}\\).

<h3>Gini index</h3>

$$\abc\$$

## Regression measures of impurity

