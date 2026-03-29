# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:14:16 2023

@author: rslay01
"""

# Importing necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

#%%
# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
#%%
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#%%
# Initialize and train the Logistic Regression classifier
clf1 = LogisticRegression()
clf1.fit(X_train, y_train)
#%%
y_pred_prob1 = clf1.predict_proba(X_test)[:, 1]
#%%
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_prob1)
auc1 = roc_auc_score(y_test, y_pred_prob1)
#%%
# Initialize and train the Random Forest classifier
clf2 = RandomForestClassifier()
clf2.fit(X_train, y_train)
y_pred_prob2 = clf2.predict_proba(X_test)[:, 1]
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_prob2)
auc2 = roc_auc_score(y_test, y_pred_prob2)
#%%
# Plotting the ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr1, tpr1, label=f'Logistic Regression (AUC = {auc1:.2f})')
plt.plot(fpr2, tpr2, label=f'Random Forest (AUC = {auc2:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

#%%
# Calculate the G-mean for each threshold
gmeans = np.sqrt(tpr1 * (1 - fpr1))

# Locate the index of the largest G-mean
ix = np.argmax(gmeans)

# Print and plot the best threshold
print(f'Best Threshold: {thresholds2[ix]} with G-mean: {gmeans[ix]}')

plt.figure(figsize=(10, 8))
plt.plot(fpr1, tpr1, label=f'Logistic Regression (AUC = {auc1:.2f})')
plt.plot(fpr2, tpr2, label=f'Random Forest (AUC = {auc2:.2f})')
plt.scatter(fpr1[ix], tpr1[ix], marker='o', color='red', label=f'Best Threshold = {thresholds1[ix]:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

#%%
# Save the Model
# import pickle as pk

# with open('My_classification_model.pkl', 'wb') as file:
#     pk.dump(clf1, file)

# modelo_cargado = pk.load('My_Model.pkl')

