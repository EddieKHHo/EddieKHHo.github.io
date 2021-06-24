---
title: "Inner workings: decision trees (Part 2)"
excerpt: "My notes on the algorithm and mathematics of decision tree classifiers and regressors."
date: 2021-06-24
last_modified_at: false
header:
  overlay_image: /assets/images/06_2021/tree.jpg
  overlay_filter: 0.5
---

This post contains the second part of my notes on decision tree classifiers and regressors. The goal in this post is to used what I learned in Part 1 and construct a decision tree step-by-step. I will utilize the algorithm laid out by the `scikit-learn` documentation and check that my tree matches the tree constructed using `scikit-learn`.

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
 	<img src="/assets/images/06_2021/iris.classifier.mod.png">
	<figcaption><b>Figure 1.</b> Simplified decision tree classifier for iris dataset.</figcaption>
</figure>

This decision tree gives you all the information you need to understand how the classifier works. For root and decision nodes, the first line indicates the feature and threshold value use to perform the split, the second gives the impurity (Gini index) of the node, the third line gives the number of samples, and the last lists the number of samples in class 0 (not versicolor) and class 1 (is versicolor). Leaf nodes are basically the same, except it does not have the first line.

As a side note, we can use this information to construct a few data point to better understand how the classifier works, using `predict` and `predict_proba`. To be precise, we will make up 5 data points with the appropriate 'petal length' and 'petal width', such that they will end up in each of the 5 leaf nodes, then examine the outputs of `predict` and `predict_proba`.

```python
X_test = [[0,0,0,0],[0,0,3,0],[0,0,5,0],[0,0,3,2],[0,0,5,2]] # data points
X_pred = clf1.predict(X_test) # classifier predict
X_prob = clf1.predict_proba(X_test) # classifier class probabilities
# output table
X_results = pd.concat([pd.DataFrame(X_test), pd.DataFrame(X_prob), pd.Series(X_pred)], axis=1)
X_results.columns = X.columns.tolist()+['Prob0','Prob1','Pred']
```

| Leaf | Petal length | Petal width | Prob0 | Prob1 | Prediction |
| :--: | :----------: | :---------: | :---: | :---: | :--------: |
|  A   |      0       |      0      |   0   |   1   |     1      |
|  B   |      3       |      0      | 0.979 | 0.021 |     0      |
|  C   |      5       |      0      | 0.333 | 0.667 |     1      |
|  D   |      3       |      2      | 0.333 | 0.667 |     1      |
|  E   |      5       |      2      |   0   |   1   |     1      |

We observed exactly what we expect. `predict_proba` outputs the proportion of class 0 and class 1 at each leaf node according the their counts. `predict` outputs the prediction based on the class that is the majority at the leaf.

# Step-by-step decision tree construction

Now let's try to replicate this tree using the CART algorithm described by the `scikit-learn` documentation. For more details, you can look at Part 1 of these notes.

For each level of the tree, I will go through these steps.

1. Pick an impure node \\(m\\).
2. Split the dataset at node \\(m\\) using all possible features, \\(f\\), and all possible threshold, \\(t\\), values starting from the minimum value of the \\(f\\) to the maximum in increments of 0.1. The "left" child node will contain samples where \\(f <= t\\) and the "right" child node will contain samples where \\(f > t\\). Measure the quality of the split using the weighted sum of Gin impurities at the child nodes.
3. Split node \\(m\\) according to the split with the lowest weighted Gini (i.e. the optimal split).
4. Repeat for all other impure nodes.

