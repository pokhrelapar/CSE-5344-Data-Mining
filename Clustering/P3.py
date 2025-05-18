"""
Ensure dependiences are installed if running on local

Graphs are saved to local directory

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Load dataset

wine_df = pd.read_csv("wine_data.csv")

# EDA

print("+" + "-" * 30 + "+")
print("| DataFrame Information:    |")
print("| Shape of wine data:", format(wine_df.shape))
print("+" + "-" * 30 + "+")

null_counts = wine_df.isnull().sum()
print("\n\n Null value counts per column:\n", null_counts)

has_nulls = wine_df.isnull().values.any()
print("\nDataFrame contains any null values:", has_nulls)

print("+" + "-" * 30 + "+")

label_counts = wine_df["quality"].value_counts()
print(label_counts)


plt.figure(figsize=(8, 6))
sns.histplot(wine_df["quality"], kde=True)
plt.title("Distribution of Wine Quality Labels")
plt.xlabel("Quality (0 or 1)")
plt.ylabel("Count")
plt.savefig("wine_quality_histogram.png")

"""## Separate features and target label"""

X = wine_df.drop("quality", axis=1)
y = wine_df["quality"]

"""## Split train and test data"""

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

"""## Normalize Dataset"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


"""## K- Nearest Neigbors

### Visualize Data

### Elbow Method

####  Identifying Optimal Value of K
"""

k_range = range(1, 40)
accuracies = []


for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker="o", linestyle="-")
plt.title("Elbow Method For Optimal k in KNN")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy Rate")
plt.xticks(np.array(k_range))
plt.grid(True)
plt.savefig("accuracy_chart.png")

k_range = range(1, 40)
errors = []


for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    errors.append(1 - accuracy)

plt.figure(figsize=(10, 6))
plt.plot(k_range, errors, marker="o", linestyle="-")
plt.title("Elbow Method For Optimal k in KNN")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Error Rate")
plt.xticks(np.array(k_range))
plt.grid(True)
plt.savefig("error_chart.png")

k_range = range(1, 40)
errors = []
accuracies = []


for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)
    errors.append(1 - accuracy)

plt.figure(figsize=(10, 6))
plt.plot(k_range, errors, marker="o", linestyle="-", label="Error Rate")
plt.plot(k_range, accuracies, marker="o", linestyle="-", label="Accuracy")
plt.title("Performance vs. Number of Neighbors (k) in KNN")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Error Rate / Accuracy")
plt.xticks(np.array(list(k_range)))
plt.legend()
plt.grid(True)
plt.savefig("performance_chart.png")

print("+" + "-" * 40 + "+")
print("|Train/Val/Test Statistics show below here: |")

print("+" + "-" * 40 + "+")

"""From the graphs I deduce the optimal value of n_neighbors as 13"""

print("+" + "-" * 65 + "+")
print("| From the graphs I deduced the optimal value of n_neighbors as 13: |")

print("+" + "-" * 65 + "+")

print("+" + "-" * 30 + "+")
print("| KNN with n = 13 neighbors    |")

print("+" + "-" * 30 + "+")

knn_optimal = KNeighborsClassifier(n_neighbors=13)
knn_optimal.fit(X_train, y_train)

train_pred_new = knn_optimal.predict(X_train)
val_pred_new = knn_optimal.predict(X_val)

train_accuracy_new = accuracy_score(y_train, train_pred_new)
val_accuracy_new = accuracy_score(y_val, val_pred_new)

print("+" + "-" * 50 + "+")
print("\tTrain Accuracy:", train_accuracy_new)
print("+" + "-" * 50 + "+")

print("+" + "-" * 50 + "+")
print("\tValidation Accuracy:", val_accuracy_new)
print("+" + "-" * 50 + "+")

print("+" + "-" * 50 + "+")
train_cm_new = confusion_matrix(y_train, train_pred_new)

print("\tConfusion Matrix for Training Data:\n")
print(pd.crosstab(y_train, train_pred_new, rownames=["Actual"], colnames=["Predicted"]))
print("+" + "-" * 50 + "+")

plt.figure(figsize=(6, 5))
sns.heatmap(
    train_cm_new,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap for Training")
plt.savefig("train_confusion_matrix.png")

val_cm_new = confusion_matrix(y_val, val_pred_new)

print("+" + "-" * 50 + "+")
print("\tConfusion Matrix for Validation Data:\n")
print(pd.crosstab(y_val, val_pred_new, rownames=["Actual"], colnames=["Predicted"]))
print("+" + "-" * 50 + "+")

plt.figure(figsize=(6, 5))
sns.heatmap(
    val_cm_new,
    annot=True,
    fmt="d",
    cmap="Greens",
    cbar=True,
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap for Validation")
plt.savefig("val_confusion_matrix.png")

"""### Dummy Data"""

"""
  Replace this with actual test data file
"""

# replace this with actual test data file
dummy_pd = pd.read_csv("wine_data_test.csv")
dummy_pd.head()

X_dummy = dummy_pd.drop("quality", axis=1)
y_dummy = dummy_pd["quality"]

X_dummy = scaler.transform(X_dummy)

test_pred_dummy = knn_optimal.predict(X_dummy)
test_accuracy_dummy = accuracy_score(y_dummy, test_pred_dummy)

print("+" + "-" * 50 + "+")
print("Test Accuracy on Dummy:", test_accuracy_dummy)
print("+" + "-" * 50 + "+")

test_dummy_cm = confusion_matrix(y_dummy, test_pred_dummy)

print("+" + "-" * 50 + "+")
print("\tConfusion Matrix for Dummy Data:\n")
print(pd.crosstab(y_dummy, test_pred_dummy, rownames=["Actual"], colnames=["Predicted"]))
print("+" + "-" * 50 + "+")

plt.figure(figsize=(6, 5))
sns.heatmap(
    test_dummy_cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    cbar=True,
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap for Dummy Data")
plt.savefig("dummy_confusion_matrix.png")

"""### Actual Testing on KNN classifier

## K-Fold Stratified Cross Validation
"""

# Load dataset
data = pd.read_csv("wine_data.csv")
X = data.drop("quality", axis=1)
y = data["quality"]

clf = KNeighborsClassifier(n_neighbors=13)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")

# accuracy for each fold.
print("10-Fold Stratified Cross-Validation Accuracies:")
for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

# average accuracy across all folds.
average_accuracy = cv_scores.mean()
print(f"\nAverage Cross-Validation Accuracy: {average_accuracy:.4f}")
