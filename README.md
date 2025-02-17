# Supervised-Techniques--Data-Science

This project implements and evaluates two machine learning classifiers—Decision Tree and Gaussian Naive Bayes—for binary classification of medical diagnosis data. The implementation includes custom-built classifiers that incorporate information gain for decision trees and Gaussian probability calculations for Naive Bayes.

Additionally, the project explores feature selection, dimensionality reduction using Singular Value Decomposition (SVD), and data augmentation using SMOTE (Synthetic Minority Over-sampling Technique) to enhance model performance and robustness on imbalanced datasets.

Implementation Details - 

1. Decision Tree Classifier (Using Information Gain)
A custom Decision Tree Classifier is implemented from scratch, following these key steps:
Entropy Calculation: Computes weighted entropy to handle class imbalance.
Information Gain Calculation: Determines the best feature to split the dataset by iteratively selecting the feature with the highest information gain.
Threshold-Based Splitting: Continuous-valued features are split based on a computed threshold (midpoint of consecutive sorted values).
Tree Construction: Recursively builds the tree until a stopping criterion (max depth or pure class labels) is met.
Prediction: Samples are classified by traversing the tree from root to leaf.
Key Features:
Class Imbalance Handling: Weighted entropy prioritizes the minority class.
Handling Continuous Features: Uses threshold-based splitting for numerical features.
Overfitting Control: Limits tree depth to 5 to enhance generalizability.
Single Sample Prediction: Supports real-time classification.

2. Gaussian Naive Bayes Classifier
A custom Gaussian Naive Bayes classifier is implemented using the following steps:
Prior Probability Calculation: Computes class priors based on data distribution.
Gaussian Likelihood Calculation: Uses the probability density function to model feature distributions.
Posterior Probability Calculation: Multiplies priors and likelihoods to obtain final probabilities.
Prediction: Selects the class with the highest posterior probability.
Key Features:
Feature Scaling: Uses StandardScaler to ensure features follow a normal distribution.
Numerical Stability: Adds a small epsilon value to prevent division by zero.

3. Dimensionality Reduction with Singular Value Decomposition (SVD)
SVD is applied to reduce feature space while preserving key variance components.
Models are trained on datasets with different ranks (reduced dimensions) to evaluate their ability to generalize with lower-dimensional representations.
Insights from SVD help determine optimal feature representation while maintaining classification accuracy.

4. Feature Selection Using Randomization
Shuffling Feature Columns: Individual feature columns are shuffled to analyze their impact.
Feature Importance Calculation: The F1-score drop after shuffling determines feature importance.
Feature Ranking: Features are ranked based on their contribution to model performance.
Visualization: Feature importance is visualized using bar charts.

5. Data Augmentation with SMOTE (Synthetic Minority Over-sampling Technique)
Synthetic Sample Generation: Creates new samples for the minority class using k-nearest neighbors.
Experimenting with Different Sampling Ratios:
Oversampling at 100%, 200%, and 300%.
Varying k_neighbors between 1 and 5.
Classifier Performance Evaluation: Models are trained and evaluated on SMOTE-augmented datasets.
Performance Metrics: Accuracy, Precision, Recall, and F1 Score are calculated to assess model improvements.
