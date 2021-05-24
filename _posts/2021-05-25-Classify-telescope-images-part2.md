---
title: "Classifying telescope image data (Part 2)"
excerpt: "Supervised machine learning with telescope image data of gamma rays."
date: 2021-05-25
last_modified_at: false
header:
  overlay_image: /assets/images/05_2021/star_galaxy_1200x777.jpg
  overlay_filter: 0.5
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"

---

I concluded in [Part 1]({% post_url 2021-05-15-Classify-telescope-images-part1 %}){:target="_blank"} of this analysis with four types of classifiers, each optimized by fine-tuning hyperparameters. From these results, it is clear that the random forest and neural network classifier are preferable to nearest neighbor and decision tree.

|    Classifier    | Training F1 | Test F1 | Test accuracy |
| :--------------: | :---------: | :-----: | :-----------: |
| Nearest neighbor |    0.736    |  0.743  |     0.843     |
|  Decision tree   |    0.762    |  0.772  |     0.850     |
|  Random forest   |    0.809    |  0.817  |     0.880     |
|  Neural network  |    0.804    |  0.811  |     0.878     |

## The Decision Threshold

Tuning the input is essential for building a good classifier, but it is equally important to study and tune the output. The output of a classifier is typically the **probability** assessment of a class. This probability is then used to bin the data point into a class based on a **decision threshold**. In most classifiers the default decision threshold is set at *t* = 0.5, such that the data point is classified as '0' if its class probability is below 0.5 and classified as '1' otherwise. 

It should be obvious now that the value of *t* can have large consequences on the predicted class, especially when class probabilities are noisy and hover around 0.5. This would impact whether the prediction is a false positive (**FP**), false negative (**FN**), true positive (**TP**), or true negative (**TN**). The count of these prediction types can be displayed on a **confusion matrix**.

|                     | Predict positive | Predict negative |
| :-----------------: | :--------------: | :--------------: |
| **Actual positive** |        TP        |        FN        |
| **Actual negative** |        FP        |        TN        |

### Example

Below is an example of how *t* may or may not affect predictions. The *Probability* and *Actual* columns represent the class probability output by the classifier and the actual class of the data point, respectively. The *Prediction-0.5* and *Result-0.5* columns represent the prediction and result when setting the decision threshold at *t* = 0.5 (analogous for *Prediction-0.6* and *Result-0.6*). In the first four rows, adjusting *t* has no effect because the class probabilities are either below 0.5 or above 0.6. In the fifth row, adjusting *t* to 0.6 changes the result from being a **FP to a TN**. In the sixth row, adjusting *t* to 0.6 changes the result from a **TP to a FN**.

| Probability | Actual | Prediction-0.5 | Prediction-0.6 | Result-0.5 | Result -0.6 |
| :---------: | :----: | :------------: | :------------: | :--------: | :---------: |
|    0.82     |   1    |       1        |       1        |     TP     |     TP      |
|    0.07     |   0    |       0        |       0        |     TN     |     TN      |
|    0.68     |   0    |       1        |       1        |     FP     |     FP      |
|    0.22     |   1    |       0        |       0        |     FN     |     FN      |
|  **0.58**   |   0    |       1        |       0        |     FP     |     TN      |
|  **0.55**   |   1    |       1        |       0        |     TP     |     FN      |

### FPR and TPR

Rather than counting each type of prediction in the confusion matrix, it is typical to calculate the false positive rate (**FPR**) and true positive rate (**TPR**). The FPR represents the proportion of actual negative events that were correctly predicted as negative events: FPR = FP /(FP+TN). The TPR represents the proportion of actual positive events that were correctly predicted as positive events: TPR = TP / (FN+TP).

Depending on the needs of your project, you can tune the decision threshold, *t*,  and optimize for the FPR and/or TPR. For example, in disease detection, you may want to maximize the TPR (i.e. reduce the FNR = 1 - TPR) because missing a positive detection can be much more consequential then falsely detecting disease presence (which would just warrant a second test to check). 

Below, I will examine the effects of adjust *t* on each of our four classifiers. In an actual study on gamma ray detection, one may prefer to optimize for low FPR at the cost of reduced TPR because there are typically many more hadronic events than gamma ray events in real data. Imagine that there are 10,000 hadronic events and only 100 gamma events. If an adjustment in *t* causes a reduction in FPR and TPR by 1%, this would reduce 100 false positives at the cost of 1 true positive.



## Python packages and functions

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score,  StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve
```

```python
def adjPred(Y_proba, t):
    """
    Adjusts class predictions based on the decision threshold (t)
    """
    return [1 if y >= t else 0 for y in Y_proba[:,1]]

def calcFprTpr(Y_test, Y_proba, t):
    '''
    Get FPR and TPR given decision threshold
    '''
    Y_pred = adjPred(Y_proba, t) #get adjusted predictions
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel() #get confusion matrix metrics
    fpr, tpr = fp/(fp+tn), tp/(tp+fn)
    return [tn, fp, fn, tp, fpr, tpr]

def tableFprTpr(Y_test, Y_proba, listT):
    '''
    Generate table of FPR, TPR with user-defined sequence of thresholds
    '''
    dfRates = pd.DataFrame(columns=['Threshold','TN','FP','FN','TP','FPR','TPR'])
    for t in listT:
        tn, fp, fn, tp, fpr, tpr = calcFprTpr(Y_test, Y_proba, t)
        dfRates.loc[len(dfRates)] = [t, tn, fp, fn, tp, fpr, tpr]
    return dfRates

def thresholdGivenFpr(Y_test, Y_proba, maxFPR):
    '''
    Find the largest decision threshold (t) that returns FPR <= maxFPR
    '''
    for t in list(np.linspace(0,1,1001)):
        tn, fp, fn, tp, fpr, tpr = calcFprTpr(Y_test, Y_proba, t)
        if fpr <= maxFPR:
            return [tn, fp, fn, tp, fpr, tpr, t]
```



## Define classifiers

Read the csv data file

```python
Telescope = pd.read_csv('telescope.csv')
```

Encode class identifier (Y), split the data into training and test set, and standardize features (X).

```python
# Define X as features and Y as labels
X = Telescope.drop('class', axis=1)
Y = Telescope['class']
# Encode Y
le = LabelEncoder()
Y_encode = le.fit_transform(Y)
# split to training and testing set, stratify by Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encode, test_size=0.30, stratify=Y)
# Standardize X_train, then fit to X_test
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Define the four classifiers based on the fine-tuned hyperparameters from [Part 1]({% post_url 2021-05-15-Classify-telescope-images-part1 %}){:target="_blank"}. Fit each classifier with training data. Obtain predictions using the default decision threshold (*t* = 0.5). Obtain the class probabilities for each data point using `predict_proba`. These class probabilities will be used to make predictions when we alter the decision threshold, *t*.

```python
#####-----nearest neighbors
NN = KNeighborsClassifier(p=1, weights='distance', n_neighbors=5)
NN.fit(X_train, Y_train)
NN_Y_pred = NN.predict(X_test)
NN_Y_proba = NN.predict_proba(X_test)

#####-----decision tree
DT = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=60, min_samples_leaf=10)
DT.fit(X_train, Y_train)
DT_Y_pred = DT.predict(X_test)
DT_Y_proba = DT.predict_proba(X_test)

#####-----random forest
RF = RandomForestClassifier(criterion='gini', max_features='sqrt', max_depth=40, n_estimators=300,min_samples_split=5, min_samples_leaf=2)
RF.fit(X_train, Y_train)
RF_Y_pred = RF.predict(X_test)
RF_Y_proba = RF.predict_proba(X_test)

#####-----neural network
MLP = MLPClassifier(activation='tanh', solver='adam', hidden_layer_sizes=(20,20), max_iter=1000, learning_rate='adaptive', alpha=0.0001)
MLP.fit(X_train, Y_train)
MLP_Y_pred = MLP.predict(X_test)
MLP_Y_proba = MLP.predict_proba(X_test)
```

## Varying the decision threshold

Before we attempt to optimize the classifiers, lets examine the effects of altering the decision threshold, *t*, on the false positive rate (FPR) and true positive rates (TPR). Although this data is usually represented in a ROC-curve (below), plotting FPR and TPR against values of *t* helped me understand this process a bit more easily.

To classify data points in the test set using a custom decision threshold, I utilize `adjPred`  (defined above), which categorized the data point as '0' if the class probability is below *t*, and as '1' otherwise.

```python
listI, listJ, listTitle =[0,0,1,1], [0,1,0,1], ['Nearest neighbor','Decision tree','Random forest','Neural network']
listT, listProba = list(np.linspace(0,1,501)), [NN_Y_proba, DT_Y_proba, RF_Y_proba, MLP_Y_proba]

# loop to produce subplots
fig, ax =plt.subplots(2, 2, figsize=(26,20))
fig.subplots_adjust(hspace=0.2, wspace=0.1)
for k in range(4):
    # table of FPR, TPR for range of decision thresholds
    model_rates = tableFprTpr(Y_test, listProba[k], listT)
    model_rates_melt = pd.melt(model_rates, id_vars=['Threshold'], value_vars=['FPR', 'TPR'], var_name='Type', value_name='Rate')
    sns.lineplot(
        data=model_rates_melt, x='Threshold', y='Rate', hue='Type', style='Type', 
        palette="colorblind", markers=True, ax=ax[listI[k],listJ[k]])
    ax[listI[k],listJ[k]].vlines(x=0.5, ymin=0, ymax=1, color='black', linestyle='--', alpha=0.8)
    ax[listI[k],listJ[k]].set_title(listTitle[k], fontsize=20, loc='left')
    ax[listI[k],listJ[k]].set_xlabel(xlabel='Threshold',fontsize=20)
    ax[listI[k],listJ[k]].set_ylabel(ylabel='Rate', fontsize=20)
    ax[listI[k],listJ[k]].tick_params(axis='both', which='major', labelsize=18)
    ax[listI[k],listJ[k]].legend(loc='upper right', prop={'size': 30})
plt.show()
```

<figure>
 	<img src="/assets/images/05_2021/RateThreshold.png">
	<figcaption><b>Figure 1.</b> FPR and TPR for range of decision thresholds. Dashed line represents the default threshold at <em>t</em> = 0.5.</figcaption>
</figure>

1. When t = 0; FPR = 0
