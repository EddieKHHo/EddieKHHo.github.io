---
title: "Markup: ipython nbconvert output"
excerpt: "This is the excerpt. Test markdown"
date: 2021-05-11
last_modified_at: false
toc : true
 toc_sticky: true
# classes: wide
---

## Table of contents
1. [Introduction](#first)
2. [Data exploration](#second)
    1. [Plot of counts](#second-sub)
3. [Another section](#third)

<a name="first"></a>
## Signals from MAGIC telescope 
This dataset was generated from a Monte Carlo program that simulates registration of high energy gamma particles detected by ground-based Major Atmospheric Gamma Imaging Cherenkov telescopes (MAGIC). When gamma rays (high energy photons) hit the Earth's atmosphere, they interact with the atoms and molecules of the air to create a particle shower, producing blue flashes of light called Cherenkov radiation. This light is collected by the telescope's mirror system and creates images of elongated ellipses.  With this data, researchers can draw conclusions about the source of the gamma rays and discover new objects in our own galaxy and supermassive black holes outside of it. However, these events are also produced by hadron showers (atomic, not photonic) from cosmic rays. The goal is to utilize features of the ellipses to separate images of gamma rays (signal) from images of hadronic showers (background).

This dataset contains 10 features from image data of **gamma (g)** and **hadronic (h)** events.

| Feature  | Description                                           |
| -------- | ----------------------------------------------------- |
| fLength  | Major axis of ellipse (mm)                            |
| fWidth   | Minor axis of ellipse (mm)                            |
| fSize    | 10-log of sum of content of all pixels                |
| fConc    | Ratio of sum of two highest pixels over fSize [ratio] |
| fConc1   | Ratio of highest pixel over fSize [ratio]             |
| fAsym    | Distance from highest pixel to center [mm]            |
| fM3Long  | 3rd root of third moment along major axis [mm]        |
| fM3Trans | 3rd root of third moment along minor axis [mm]        |
| fAlpha   | Angle of major axis with vector to origin [deg]       |
| fDist    | Distance from origin to center of ellipse [mm]        |

I will utilize this dataset to explore several supervised machine learning algorithms aimed at classifying the events as gamma or hadronic based on these 10 features. My goal here is mainly to gain some practice utilizing nearest neighbor, decision tree, random forest and neural networks for classification. 


<a name="second"></a>
## Data exploration

Iimport from NumPy, Pandas, MatplotLib, Seaborn and Scikit-learn

```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score,  StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer, classification_report, confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve
```
Read the csv data file.

```python
Telescope = pd.read_csv('telescope.csv')
```
Examining the data structure, we see that all the features are floats and the class (g or h) is an object type.

```python
Star.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19020 entries, 0 to 19019
Data columns (total 11 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   fLength   19020 non-null  float64
 1   fWidth    19020 non-null  float64
 2   fSize     19020 non-null  float64
 3   fConc     19020 non-null  float64
 4   fConc1    19020 non-null  float64
 5   fAsym     19020 non-null  float64
 6   fM3Long   19020 non-null  float64
 7   fM3Trans  19020 non-null  float64
 8   fAlpha    19020 non-null  float64
 9   fDist     19020 non-null  float64
 10  class     19020 non-null  object 
dtypes: float64(10), object(1)
```
Since this is simulated data, there are no missing values.
```python
Telescope.isnull().sum()
```
```
fLength     0
fWidth      0
fSize       0
fConc       0
fConc1      0
fAsym       0
fM3Long     0
fM3Trans    0
fAlpha      0
fDist       0
class       0
dtype: int64
```
Describing the data, we observe that the features have very different ranges. This is not surprising, given that some features are ratios and others are 3rd root of values. It will be important to standardize them prior to classification.
```python
Telescope.describe().loc[['count','mean','std','min','50%','max']]
```
```
            fLength        fWidth         fSize         fConc        fConc1  \
count  19020.000000  19020.000000  19020.000000  19020.000000  19020.000000   
mean      53.250154     22.180966      2.825017      0.380327      0.214657   
std       42.364855     18.346056      0.472599      0.182813      0.110511   
min        4.283500      0.000000      1.941300      0.013100      0.000300   
25%       24.336000     11.863800      2.477100      0.235800      0.128475   
50%       37.147700     17.139900      2.739600      0.354150      0.196500   
75%       70.122175     24.739475      3.101600      0.503700      0.285225   
max      334.177000    256.382000      5.323300      0.893000      0.675200   

              fAsym       fM3Long      fM3Trans        fAlpha         fDist  
count  19020.000000  19020.000000  19020.000000  19020.000000  19020.000000  
mean      -4.331745     10.545545      0.249726     27.645707    193.818026  
std       59.206062     51.000118     20.827439     26.103621     74.731787  
min     -457.916100   -331.780000   -205.894700      0.000000      1.282600  
25%      -20.586550    -12.842775    -10.849375      5.547925    142.492250  
50%        4.013050     15.314100      0.666200     17.679500    191.851450  
75%       24.063700     35.837800     10.946425     45.883550    240.563825  
max      575.240700    238.321000    179.851000     90.000000    495.561000  
```

### Number of hadron and gamma signals
There is a bias towards a higher number of data points for gamma (12332) events than hadronic (6688) events. This suggests that we should use a machine learning score that is less senstitive to this type of bias, such as the F1 score.
```python
Star['class'].value_counts()
```
```
g    12332
h     6688
Name: class, dtype: int64
```
```python
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=Star, x='class')
ax.set_xlabel('Class', fontsize=20)
ax.set_ylabel('Count', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('images/class.count.png')
plt.show()
```
<figure>
 	<img src="/assets/images/05_2021/class.count.png">
	<figcaption>Counts of gamma (g) and hadronic (h) events.</figcaption>
</figure>


<a name="third"></a>
### Pair plot of all features

```python
sns.set_context("paper", rc={"axes.labelsize":20})
pp = sns.pairplot(Telescope, height = 2.5, hue='class', plot_kws={'alpha':0.5})
pp._legend.remove()
```
<figure>
 	<img src="/assets/images/05_2021/features.pairplot.png">
	<figcaption>Pair plot for all features. Gamma and hadronic are in blue and orange, respectively.</figcaption>
</figure>


### Boxplot of features separated by class
We observe that gamma and hadronic events have similar distributions for most individual features. The one exception is *fAlpha*, where the hadronic events tend to have larger values. Regardless, this indicates that any individual feature is unliely to strongly differentiate gamma from hadronic events. 
```python
sns.set_context("paper", rc={"axes.labelsize":20})
fig, ax =plt.subplots(2, 5, figsize=(26,20))
sns.boxplot(x='class', y='fLength', data=Telescope, ax=ax[0,0])
sns.boxplot(x='class', y='fWidth', data=Telescope, ax=ax[0,1])
sns.boxplot(x='class', y='fSize', data=Telescope, ax=ax[0,2])
sns.boxplot(x='class', y='fConc', data=Telescope, ax=ax[0,3])
sns.boxplot(x='class', y='fConc1', data=Telescope, ax=ax[0,4])
sns.boxplot(x='class', y='fAsym', data=Telescope, ax=ax[1,0])
sns.boxplot(x='class', y='fM3Long', data=Telescope, ax=ax[1,1])
sns.boxplot(x='class', y='fM3Trans', data=Telescope, ax=ax[1,2])
sns.boxplot(x='class', y='fAlpha', data=Telescope, ax=ax[1,3])
sns.boxplot(x='class', y='fDist', data=Telescope, ax=ax[1,4])
fig.show()
```
<figure>
 	<img src="/assets/images/05_2021/features.boxplot.png">
	<figcaption>Box plot for all features. Gamma and hadronic are in blue and orange, respectively.</figcaption>
</figure>


### Principal components analysis
For classification problems, I typically perform a PCA to reduce dimensionality and see if the classes can be separated visually. 
Here, we see that the first and second principal components explain 42% and 16% of the variance.

```python
#####-----Get features and standardize using StandardScaler()
scaler = StandardScaler()
X = Telescope.drop('class', axis=1)
X_full_scaled = scaler.fit_transform(X)
#####-----Perform PCA with 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_full_scaled)
print(pca.explained_variance_ratio_) 
```
```
[0.42239909 0.15751879]
```
However, the gamma and hadronic events do not seem to cluster separately on these PCs. A lot of hadronic events cluster on the top-left side of the plot, but the majority clusters with the gamma events.
```python
#####-----Create dataframe for plotting
dfPC = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
dfFinal = pd.concat([dfPC, Telescope[['class']]], axis = 1)
#####-----Plot data of PC axes
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
targets = ['g','h']
colors = ['blue','orange']
for target, color in zip(targets,colors):
    indicesToKeep = dfFinal['class'] == target
    ax.scatter(dfFinal.loc[indicesToKeep, 'principal component 1'],
               dfFinal.loc[indicesToKeep, 'principal component 2'],
               c=color, s=50, alpha=0.25)
ax.set_xlabel('Principal Component 1 (42%)', fontsize=20)
ax.set_ylabel('Principal Component 2 (16%)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(targets, prop={'size': 15})
ax.grid()
```
<figure>
 	<img src="/assets/images/05_2021/pca.png">
	<figcaption>Data points on PC1 and PC2. Gamma and hadronic are in blue and orange, respectively.</figcaption>
</figure>

## Data preprocessing

Here I prepare the data for supervised machine learning. First I split the dataset in features *X* and classes *Y*. I will utilize `LabelEncoder` to convert *g* and *h* into binary format.

```python
# Define X as features and Y as labels
X = Telescope.drop('class', axis=1)
Y = Telescope['class']
# Encode Y
le = LabelEncoder()
Y_encode = le.fit_transform(Y)
print(pd.crosstab(Y, Y_encode, rownames=['Original'], colnames=['Encode']))
```

```
Encode        0     1
Original             
g         12332     0
h             0  6688
```

Now, split the data into the training and the test set using `train_test_split`. I will make the training set be 70% of the data and stratify by *Y* because the number of gamma and hadronic events are unbalanced.

Lastly, I standardize each feature using `StandardScaler` because they vary a lot in their ranges. Remember to only utilize the training set when fitting the scaler, that way the no information from the test set will leak into the training set.

```python
# split, stratify by Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encode, test_size=0.30, stratify=Y)

# Standardize X_train, then fit to X_test
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

## Supervised machine learning

Here I explore using nearest neighbors, decision tree, random forest, and neural networks to classify the data. Since the number of gamma and hadron events are unbiased, I will focus on using the F1 score to determine the fit of the model. For each classifier, I will first perform optimization using `GridSearchCV` or `RandomizedSearchCV` to find the hyperparameters that maximize F1 on the training set. This will be followed by some exploratory analysis on the effects of select hyperparameters when performing cross-validation on the training set and ultimately when making predictions for the test set.

Below are just some wrapper functions used for fine-tuning hyperparameters.

```python
def grid_search_wrapper(model, parameters, cv, scoring):
    '''
    Perform grid search given model and parameters
    Outputs parameters and scores for best model
    '''
    ##########----------Perform grid search
    model_GS = GridSearchCV(model, parameters, cv=cv, scoring=scoring, verbose=1, n_jobs = 3)
    model_GS.fit(X_train, Y_train)
    print('GS best parameters: ',model_GS.best_params_)
    print('GS best score: ',model_GS.best_score_)
    ##########----------Test score
    Y_pred = model_GS.predict(X_test)
    print("Confusion matrix:\n", confusion_matrix(Y_test, Y_pred))
    print("F1:", f1_score(Y_test, Y_pred))
    print("Accuracy:", accuracy_score(Y_test, Y_pred))

def random_search_wrapper(model, parameters, cv, scoring, n_iter):
    '''
    Perform randomized search given model and parameters
    Outputs parameters and scores for best model
    '''
    ##########----------Perform grid search
    model_GS = RandomizedSearchCV(model, parameters, cv=cv, scoring=scoring, n_iter=n_iter, verbose=1, n_jobs = 3)
    model_GS.fit(X_train, Y_train)
    print('GS best parameters: ',model_GS.best_params_)
    print('GS best score: ',model_GS.best_score_)
    ##########----------Test score
    Y_pred = model_GS.predict(X_test)
    print("Confusion matrix:\n", confusion_matrix(Y_test, Y_pred))
    print("F1:", f1_score(Y_test, Y_pred))
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
```



### Nearest neighbors classifier



