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

It is useful to first learn about the different parts of a decision tree. At the top is the **root** node, which is the starting point and represents the entire dataset. At each step, the dataset is **split** into two (or more) sub-nodes based on the particular value of one of its features. Which features to utilize and what values to split the data points on is the main challenge on constructing the model. There are numerous algorithms that can be used and one of the most popular is the **classification and regression tree (CART)** algorithm, which will be discussed in detail in the rest of these notes. After a split, the resulting sub-nodes can either be a **leaf (terminal)** node or a **decision** node. The leaf node represents an end to a tree and will not be subject to further splitting. Leaf nodes may be nodes that have just one type of target variable and so no further splitting is required or there could be restrictions to the depth of the tree causing all nodes after a certain number of splits to become leaves. Decision nodes represents one where further splitting will occur. You can actually think of the root node as just the first decision node of the tree.

<figure>
 	<img src="/assets/images/06_2021/DecisionTree.png">
	<figcaption><b>Figure 1.</b> Diagram of a simple decision tree.</figcaption>
</figure>
# Building a decision tree

1. Begin with the entire dataset at the root node.
2. Search over all possible splits and find the one that is most informative about the target; each split only concerns one feature. A split typically involves splitting the data base on whether its feature value is <= or > than a **threshold** value. The aim is to find the split that minimized the **impurity** of the sub-nodes.
3. Use the most informative split (minimum impurity) to divide the dataset into "left" and "right" sub-nodes.
4. Repeat steps 2 and 3 to recursively partition the dataset until each leaf only contains a single target class / regression value. A leaf node containing only data points with the same target value is a called **pure**. However, a leaf node can be impure for at least two reasons. Firstly, there may simply not be enough information in the features to separate  data points with different target values. Secondly, construction of the tree may be halted if some restriction is met to reduce overfitting (e.g. maximum depth was reached).

## Prediction

Once a decision tree is learned from training on the data, a prediction is made for new data points by looking at which leaf node they fall on based on their feature values. The prediction for the leaf node is simply the **majority target** for decision tree classifiers or the **mean value of targets** for decision tree regressors. Note that for decision tree regressors, the predicted value will never be outside of the values in the training dataset (i.e. it does not extrapolate). 

## Overfitting

Building a tree until all leaves are pure usually causes the model to overfit onto the training dataset. To prevent overfitting, one can perform **pre-pruning** to stop building the tree early on or **post-pruning** to remove or collapse nodes with little information. Some examples of pre-pruning include limiting the max depth of the three, limiting the maximum number of leaves, and requiring a minimum number of data point to continue splitting a node.

# Measure of impurity

Impurity of node

Impurity of split.

## Classification measures of impurity



## Regression measures of impurity

