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

