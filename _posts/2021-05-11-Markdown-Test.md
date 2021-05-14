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
##########----------Read telescope data as Star
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
Describing the data, we observe that the features have very different ranges. This is not surprising, given that some features are rations and others are 3rd root of values. It will be important to standardize them prior to classification.
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
<figure class="half">
	<img src="/assets/images/TestConvert_4_1.png">
  <img src="/assets/images/class.count.png">
	<figcaption>Figure caption.</figcaption>
</figure>

<a name="third"></a>
## Third section

