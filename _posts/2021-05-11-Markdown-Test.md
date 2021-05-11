---
title: "Markup: ipython nbconvert output"
excerpt: "This is the excerpt. Test markdown"
date: 2021-05-11
last_modified_at: false
#toc : true
# toc_sticky: true
classes: wide
---

## Table of contents
1. [Introduction](#first)
2. [Target counts](#second)
    1. [Plot of counts](#second-sub)
3. [Another section](#third)

<a name="first"></a>
## Classifying signals from MAGIC telescope 
Some description text

```python
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, scale, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```


```python
##########----------Read telescope data as Star
Star = pd.read_csv('telescope.csv')
```


```python
##########----------All features are numeric except for class
print('-'*50)
print(Star.info())
print('-'*50)
print(Star.describe())
##########---------There are 0 NAs
print('-'*50)
print(Star.isnull().sum())
```
Some output

```
--------------------------------------------------
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
memory usage: 1.5+ MB
None
--------------------------------------------------
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
--------------------------------------------------
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
<a name="second"></a>
## Number of hadron and gamma signals
Plotting a histogram for counts of hardon and gamma signals.

```python
#########----------Barplot of class counts
print(Star['class'].value_counts())
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=Star, x='class')
ax.set_xlabel('Class', fontsize=20)
ax.set_ylabel('Count', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('images/class.count.png')
plt.show()
```
```
g    12332
h     6688
Name: class, dtype: int64
```

| Type   | Count  |
| :----  | :----: |
| Gamma  | 12332  |
| Hadron | 6688   |

| Header1 | Header2 | Header3 |
|:--------|:-------:|--------:|
| cell1   | cell2   | cell3   |
| cell4   | cell5   | cell6   |
|-----------------------------|
| cell1   | cell2   | cell3   |
| cell4   | cell5   | cell6   |
|=============================|
| FOOT11   | FOOT22   | FOOT33   |

<a name="second-sub"></a>
### plot of counts
<figure class="half">
	<img src="/assets/images/TestConvert_4_1.png">
  <img src="/assets/images/class.count.png">
	<figcaption>Figure caption.</figcaption>
</figure>

<a name="third"></a>
## Third section

