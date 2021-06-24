---
title: "Inner workings: decision trees (Part 2)"
excerpt: "My notes on the algorithm and mathematics of decision tree classifiers and regressors."
date: 2021-06-24
last_modified_at: false
header:
  overlay_image: /assets/images/06_2021/slim.code.png
  overlay_filter: 0.5
---

This post contains the second part of my notes on decision tree classifiers and regressors. You can have a look of part 1 here. The goal in this post is to used what I learned in part 1 and construct a decision tree step-by-step. I will utilize the algorithm laid out by the `scikit-learn` documentation and check that my tree matches the tree constructed using `scikit-learn`.

# Simplified decision tree of Iris dataset

