import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             roc_curve, auc, confusion_matrix, precision_recall_curve)
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize

# Setup for plots
sns.set(style="whitegrid")

#-----------------------------------------------------------
# 1. LOADING AND EXPLORING THE DATA
#-----------------------------------------------------------
print("Loading data...")
train_data = pd.read_csv('C:/Users/subha/Desktop/Workspace/NEU/AI_Sys_Tech/Week-4/Auto_fraud_detection_dataset/train.csv')
test_data = pd.read_csv('C:/Users/subha/Desktop/Workspace/NEU/AI_Sys_Tech/Week-4/Auto_fraud_detection_dataset/test.csv')

print("Data loaded successfully.\n")

print("---- Data Exploration ----")
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print("\nSample rows from train_data:")
print(train_data.head())

print("\nStatistical summary of train_data:")
print(train_data.describe())

print("\nMissing values in train_data:")
print(train_data.isnull().sum())
print("--------------------------\n")

#-----------------------------------------------------------
# 2. DATA PREPARATION
#   (First column is the target variable)
#-----------------------------------------------------------
# Because first column is the target, we select that column as y,
# and the remaining columns as X.
y_train = train_data.iloc[:, 0]
X_train = train_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]

print("Preprocessing data...")
# Simple imputation of missing values with mean
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
print("Preprocessing completed.\n")

#-----------------------------------------------------------
# 3. MODEL TRAINING WITH HYPERPARAMETER TUNING (XGBoost)
#-----------------------------------------------------------
print("Starting model training...")
model = XGBClassifier(objective='binary:logistic', random_state=27)
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
print("Model training completed.\n")

best_model = grid_search.best_estimator_
print("Best model parameters:", grid_search.best_params_, "\n")

#-----------------------------------------------------------
# 4. INITIAL MODEL EVALUATION (Threshold = 0.5)
#-----------------------------------------------------------
print("Evaluating model at default threshold (0.5)...")
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy on test data: {:.2f}".format(accuracy))
print("F1 Score on test data: {:.2f}".format(f1))
print("Classification Report:\n", classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix (Threshold=0.5)')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.savefig('confusion_matrix_default.png')
plt.close()

#-----------------------------------------------------------
# 5. ROC CURVE
#-----------------------------------------------------------
y_test_bin = label_binarize(y_test, classes=[0,1])
y_scores = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test_bin, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()
print("ROC Curve AUC: {:.2f}\n".format(roc_auc))

#-----------------------------------------------------------
# 6. PRECISION-RECALL CURVE & THRESHOLD SELECTION
#-----------------------------------------------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = []

for p, r in zip(precision, recall):
    if (p + r) == 0:
        f1_scores.append(0)
    else:
        f1_scores.append(2 * p * r / (p + r))

best_index = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_index]
best_f1_pr = f1_scores[best_index]

print(f"Best threshold from PR curve: {best_threshold:.2f}")
print(f"F1 Score at best threshold: {best_f1_pr:.2f}\n")

plt.figure(figsize=(8,6))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.scatter(recall[best_index], precision[best_index], color='red',
            label=f'Best Threshold={best_threshold:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.savefig("precision_recall_curve.png")
plt.close()

# Evaluate model at the new threshold
print(f"Evaluating model at the new threshold={best_threshold:.2f}...")
y_pred_custom = (y_scores >= best_threshold).astype(int)

accuracy_custom = accuracy_score(y_test, y_pred_custom)
f1_custom = f1_score(y_test, y_pred_custom)

print("Accuracy at best threshold: {:.2f}".format(accuracy_custom))
print("F1 Score at best threshold: {:.2f}".format(f1_custom))
print("Classification Report:\n", classification_report(y_test, y_pred_custom))

cm_custom = confusion_matrix(y_test, y_pred_custom)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_custom, annot=True, fmt="d", cmap='Greens')
plt.title(f'Confusion Matrix (Threshold={best_threshold:.2f})')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.savefig('confusion_matrix_custom_threshold.png')
plt.close()

#-----------------------------------------------------------
# 7. FEATURE IMPORTANCE
#-----------------------------------------------------------
feature_importance = best_model.feature_importances_
indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), feature_importance[indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.savefig('feature_importances.png')
plt.close()

print("All plots saved (confusion matrices, ROC, PR curve, feature importances).")
print("Script complete. Optional: Save the model if needed.\n")
# best_model.save_model('xgboost_model.bin')
