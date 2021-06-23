---
title: "Notes on decision tree classifiers/regressors"
excerpt: "My notes on the algorithm and mathematics of decision tree classifiers and regressors."
date: 2021-07-05
last_modified_at: false
header:
  overlay_image: /assets/images/06_2021/slim.code.png
  overlay_filter: 0.5
---



## Decision tree

Decision trees are a non-parametric supervised learning method for classification and regression. The goal is to create a model that learns a hierarchy of if/else questions, using the data features to predict the value of the target variable. Decision trees are relatively simple to understand and can be easily visualized using a **decision tree**.

It is useful to first learn about the different parts of a decision tree. At the top is the **root** node, which is the starting point and represents the entire dataset. At each step, the dataset is **split** into two (or more) sub-nodes based on the particular value of one of its features. Which features to utilize and what values to split the data points on is the main component of what is learned by the model. There are numerous algorithms that can be used and one of the most popular is the classification and regression tree (CART) algorithm, which will be discussed in detail in the rest of these notes. After a split, the resulting sub-nodes can either be a **leaf (terminal)** node or a decision **node**. The leaf node represents an end to a tree and will not be subject to further splitting. Leaf nodes may be nodes that have just one type of target variable and so no further splitting is required or there could be restrictions to the depth of the tree causing all nodes after a certain number of splits to become leaves. Decision nodes represents one where further splitting will occur. You can actually think of the root node as just the first decision node of the tree.

<figure>
 	<img src="/assets/images/06_2021/DecisionTree.png">
	<figcaption><b>Figure 1.</b> Diagram of a simple decision tree.</figcaption>
</figure>

