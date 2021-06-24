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

I will go through these steps for each level of the tree:

1. Pick an impure node \\(m\\).
2. Split the dataset at node \\(m\\) using all possible features, \\(f\\), and all threshold values, \\(t\\), starting from the minimum value of \\(f\\) to the maximum in increments of 0.1. The "left" child node will contain samples where \\(f <= t\\) and the "right" child node will contain samples where \\(f > t\\). Measure the quality of the split using the weighted sum of Gini impurities at the child nodes.
3. Split node \\(m\\) according to the split with the lowest weighted Gini (i.e. the optimal split).
4. Repeat for all other impure nodes.

## Relevant functions

Functions to calculate the Gini index of a single node and the weighted Gini of a pair of left and right child nodes.

```python
def gini_index(targets):
    '''
    Calculate gini index for binary list (0, 1)
    '''
    p0 = sum([x==0 for x in targets]) / len(targets) if len(targets)>0 else 0
    p1 = 1-p0
    gini = 1 - (p0**2 + p1**2)
    return gini

def gini_weighted(targets_left, targets_right):
    '''
    Calculate weighted gini of split
    '''
    prop_left = len(targets_left) / (len(targets_left)+len(targets_right))
    prop_right = len(targets_right) / (len(targets_left)+len(targets_right))
    gini_split = prop_left*gini_index(targets_left) + prop_right*gini_index(targets_right)
    return gini_split
```

This function generates a table of the weighted Gini for all splits at the node for each specified feature. For each feature, it splits the node using threshold values starting from the minimum value to the maximum in increments of 0.1.

```python
def gini_all_splits(data, features, target):
    '''
    get results of spliting data by each feature and a range of thresholds
    '''
    # split using each feature and each threshold
    table_split = pd.DataFrame(columns=['Feature','t','gini_left','gini_right','gini_split'])
    for F in features:
        t_min, t_max = min(data[F]), max(data[F])
        step = 0.01
        t_list = list(np.arange(t_min, t_max+step, step)) # get list of thresholds
        #t_list.reverse()
        t_list = [round(x,2) for x in t_list]
        # split using each threshold and calculate gini
        for t in t_list:
            data_left = data[data[F] <= t].reset_index(drop=True)
            data_right = data[data[F] > t].reset_index(drop=True)
            gini_left = gini_index(data_left[target])
            gini_right = gini_index(data_right[target])
            gini_split = gini_weighted(data_left[target], data_right[target])
            table_split.loc[len(table_split)] = [F, t, gini_left, gini_right, gini_split]
    return table_split
```

This function takes the output of `gini_all_splits` and finds the first split with the minimum weighted Gini for each feature.

```python
def gini_best_split(data, features, target):
    '''
    obtain best split for each feature
    '''
    table_split = gini_all_splits(data, features, target) # table of all splits using all features
    # get best split for each feature
    table_best_split = pd.DataFrame()
    for F in features:
        table_feature = table_split[table_split['Feature']==F].reset_index(drop=True) # subset to focal feature
        index_min_gini = table_feature.gini_split.argmin() # get index of lowest gini_split
        best_row = table_feature.iloc[index_min_gini]
        table_best_split = table_best_split.append(best_row).reset_index(drop=True)
    return table_best_split
```

## First split

First we setup the iris dataset as above, by converting it to a binary target to indicate if the sample is 'versicolor' or not.

```python
iris = load_iris(as_frame=True)
dfIris = iris.data
# target is converted to binary
versicolor = [0 if x==1 else 1 for x in iris.target]
dfIris['versicolor'] = versicolor
# define features and target columns
features = iris.feature_names
target = 'versicolor'
```

Let's first examine the weighted Gini of all splits output by `gini_all_splits.`

```python
# get split impurity for all features and thresholds values
first_split = gini_all_splits(dfIris, features, target)
# plot
fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(
    data=first_split, x='t', y='gini_split', 
    hue='Feature', palette="colorblind", markers=True)
ax.set_xlabel(xlabel='Threshold',fontsize=20)
ax.set_ylabel(ylabel='Gini impurty of split', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(prop={'size': 15})
plt.show()
```

<figure>
 	<img src="/assets/images/06_2021/iris.first_split.png">
	<figcaption><b>Figure 2.</b> Impurity of first splits for each feature and threshold.</figcaption>
</figure>

We actually observe that there are ties for the lowest weighted Gini when splitting by 'petal length' and by 'petal width'.

Let's now look at the optimal splits for each feature output by `gini_best_split`.

```python
table_best_split = gini_best_split(dfIris, features, target)
```

| Feature      | t    | Gini_left | Gini_right | Gini_split |
| ------------ | ---- | --------- | ---------- | ---------- |
| sepal length | 5.4  | 0.204     | 0.495      | 0.394      |
| sepal width  | 2.9  | 0.481     | 0.285      | 0.360      |
| petal length | 1.9  | 0         | 0.5        | 0.333      |
| petal width  | 0.6  | 0         | 0.5        | 0.333      |

As expected, we observe a tie for the lowest weighted Gini (0.333) when splitting using 'petal length' with threshold 1.9 and 'petal width' with threshold 0.6. There are two things of note here. Firstly, we know from the plot above that there are many other threshold values that would give the same weighted Gini of 0.333. The ones shown in the table are just the ones that appear first. Second, the decision classifier created by `scikit-learn` in Figure 1, choose to split by 'petal length' with threshold 2.45, which according to Figure 2 will tie for the weighted Gini score of 0.333. 
