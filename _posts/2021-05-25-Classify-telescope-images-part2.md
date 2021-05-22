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

Tuning the input is essential for building a good classifier, but it is equally important to study and tune the output. The output of a classifier is typically the **probability** assessment of a class. This probability is then used to bin the data point into a class based on a **decision threshold**. In most classifiers the default decision threshold is set at *t* = 0.5, such that the data point is classified as '0' if its class probability is below 0.5 and classified as '1' otherwise. 

It should be obvious now that the value of *t* can have large consequences on the predicted class, especially when class probabilities are noisy and hover around 0.5. This would impact whether the prediction is a **false positive (FP)**, **false negative (FN)**, **true positive (TP)**, or **true negative (TN)**. The count of these prediction types can be displayed on a **confusion matrix**.

|                 | Predict positive | Predict negative |
| :-------------: | :--------------: | :--------------: |
| Actual positive |        TP        |        FN        |
| Actual negative |        FP        |        TN        |



Below is an example of how *t* may or may not affect predictions. The *Probability* and *Actual* columns represent the class probability output by the classifier and the actual class of the data point, respectively. The *Prediction-0.5* and *Result-0.5* columns represent the prediction and result when setting the decision threshold *t* = 0.5; analogously for *Prediction-0.6* and *Result-0.6*. In the first four rows, adjusting *t* has no effect because the class probabilities are either below 0.5 or above 0.6. In the fifth row, adjusting *t* to 0.6 changes the result from being a FP to a TN. in the sixth row, adjusting *t* to 0.6 changes the result from a TP to a FN.

| Probability | Actual | Prediction-0.5 | Prediction-0.6 | Result-0.5 | Result -0.6 |
| ----------- | ------ | -------------- | -------------- | ---------- | ----------- |
| 0.82        | 1      | 1              | 1              | TP         | TP          |
| 0.07        | 0      | 0              | 0              | TN         | TN          |
| 0.68        | 0      | 1              | 1              | FP         | FP          |
| 0.22        | 1      | 0              | 0              | FN         | FN          |
| 0.58        | 0      | 1              | 0              | FP         | TN          |
| 0.55        | 1      | 1              | 0              | TP         | FN          |



