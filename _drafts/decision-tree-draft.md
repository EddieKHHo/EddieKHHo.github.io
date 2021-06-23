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
The Gini index is used by the **CART** algorithm. It measures how often a randomly chosen data point in the node would be incorrectly labeled. A pure and a maximumally impure node will have Gini index values of 0 and  0.5, respectively.

$$\begin{aligned}
H(S_{m}) &=\sum_{k} p_{m,k}(1-p_{m,k}) \\
&=\sum_{k} p_{m,k}-\sum_{k} p_{m,k}^{2} \\
&=1-\sum_{k} p_{m,k}^{2}  \end{aligned}$$

<h3>Entropy</h3>
Entropy is a measure of disorder, which increases with increasing abundances of data points with different target values. A pure and a maximullay impure node will have entropy values of 0 and  0.5, respectively.

$$H(S_{m}) =-\sum_{k} p_{m,k}\log_{2}p_{m,k}$$

<h3>Example</h3>
If we assume that the dataset only has two classes (0 and 1). At node \\(m\\) the proportion of data of class 0 and class 1 will be \\(p_{0}\\) and \\(p_{1}=1-p_{0}\\), respectively. For this type of data, we can examine the Gini and entropy curve for all values of \\(p_{0}\\) from 0 to 1. 

```python
from math import log
import numpy as np
import pandas as pd
import seaborn as sns

def gini_index(p0):
    '''
    Calculate gini index for binary target variable
    '''
    p1 = 1-p0
    return 1 - (p0**2 + p1**2)

def entropy(p0):
    '''
    Calculate entropy for binary target variable
    '''
    p1 = 1-p0
    try:
        return (-1 * (p0 * log(p0, 2) + p1 * log(p1, 2)))
    except ValueError:
        return 0

# create lists of p0, Gini and Entropy
listP0 = list(np.arange(0,1+0.01,0.01))
listGini = [gini_index(p0) for p0 in listP0]
listEntropy = [entropy(p0) for p0 in listP0]
# create dataframe
dfImpurity = pd.DataFrame({
    'p0':listP0+listP0+listP0,
    'measure':['p0']*len(listP0)+['Gini']*len(listGini)+['Entropy']*len(listEntropy),
    'impurity':listP0+listGini+listEntropy
})
# plot using seaborn
fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(
    data=dfImpurity, x='p0', y='impurity', 
    hue='measure', palette="colorblind", markers=True)
ax.set_xlabel(xlabel='Proportion of class 0',fontsize=20)
ax.set_ylabel(ylabel='Impurity', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(prop={'size': 15})
plt.show()
```

<figure>
 	<img src="/assets/images/06_2021/impurity.gini.entropy.png">
	<figcaption><b>Figure 2.</b> Gini and entropy curve for binary class target variable. The blue line representing the proportion of class 0 is just used for reference.</figcaption>
</figure>

For both measures of impurity, there is a peak at \\(p_{0} = 0.5\\), which represents a node with maximal impurity. Impurity then approaches 0 as \\(p_{0}\\) approaches 0 and 1, which represents nodes with only class 0 or only class 1 target variables.

## Regression measures of impurity

Given a continuous target variable at node \\(m\\) with values represented by vector \\(y\\). Let \\(\bar{y}\\) as the mean and \\(\tilde{y}\\) as the median of vector y.
