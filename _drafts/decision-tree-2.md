---
title: "Inner workings: decision trees (Part 2)"
excerpt: "My notes on the algorithm and mathematics of decision tree classifiers and regressors."
date: 2021-06-24
last_modified_at: false
header:
  overlay_image: /assets/images/06_2021/tree.jpg
  overlay_filter: 0.5
---

This post contains the second part of my notes on decision tree classifiers and regressors. You can have a look of part 1 here. The goal in this post is to used what I learned in part 1 and construct a decision tree step-by-step. I will utilize the algorithm laid out by the `scikit-learn` documentation and check that my tree matches the tree constructed using `scikit-learn`.

# Packages

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_iris
```



# Simplified decision tree of Iris dataset

Here I will construct a decision tree classifier using the iris dataset. To simplify the tree, I will convert the target variable from having three classes {0: setosa, 1: versicolor, 2: virginica} to just two classes {0: not versicolor, 1: is versicolor) and limit the depth of the tree to three levels.

```python
# load iris dataset
iris = load_iris(as_frame=True)
dfIris = iris.data
# add a new target column called 'versicolor' to indicate if sample is 'versicolor' or not
versicolor = [0 if x==1 else 1 for x in iris.target]
dfIris['versicolor'] = versicolor
```

We can now construct the decision tree classifier with maximum depth of three and plot the tree using the `plot_tree` method.

```python
# define features and targets
features = iris.feature_names
target = 'versicolor'
# define parameters for tree
random_state = 111
max_depth = 3
# fit decision tree classifier
X, Y = dfIris[features], dfIris[target]
clf1 = DecisionTreeClassifier(random_state=111, max_depth=max_depth)
clf1 = clf1.fit(X, Y)
# plot tree
plot_tree(clf1, feature_names=features, filled=True)
```

<figure>
 	<img src="/assets/images/06_2021/iris.classifier.png">
	<figcaption><b>Figure 1.</b> Simplified decision tree classifier for iris dataset.</figcaption>
</figure>

This decision tree gives you all the information you need to understand how the classifier works. For root and decision nodes, the first line indicates the feature and threshold value use to perform the split, the second line indicates the impurity (Gini index) of the node, the third line indicates the number of samples, and the last line indicates the number of samples in class 0 (not versicolor) and class 1(is versicolor). Leaf nodes are basically the same, except it does not have the first line because it is not split further.
