import io
import pandas as pd
from google.colab import files

"""# Uploading File"""

from sklearn.datasets import load_iris

iris_bunch = load_iris()

iris_df = pd.DataFrame(iris_bunch.data, columns=iris_bunch.feature_names)
iris_df['species'] = iris_bunch.target

print(iris_df.head())

print(iris_df)

"""Essential Pandas Commands to inspect files"""

iris_df.info()

iris_df.describe()

iris_df.isnull().sum()

"""Split train and test data"""

X = iris_df.drop('species', axis = 1)
y = iris_df['species']

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""Scaling data"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""# SVC Model"""

model = SVC(kernel = 'linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

"""# Part B"""