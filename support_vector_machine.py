import io
import pandas as pd
from google.colab import files

uploaded = files.upload()


files = pd.read_excel("/content/Breast_Cancer_Data.xlsx")

"""Essential Pandas Commands to inspect files"""

files.info()

files.describe()

files.isnull().sum()

"""# Dataset to be used"""

dataset1 = files.drop(['id'], axis = 1, errors = 'ignore')
print(dataset1)

"""# Converting M and B into 1 and 0"""

dataset1['diagnosis'] = dataset1['diagnosis'].astype(str).str.strip().str.upper().map({'M':1, 'B':0})
display(dataset1)

"""Split train and test data"""

X = dataset1.drop('diagnosis', axis = 1)
y = dataset1['diagnosis']

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

"""# Identifying SV indices"""

support_vector_indices = model.support_
print("Indices of Support Vectors", support_vector_indices)

len(support_vector_indices)

print('No of vectors', model.n_support_)