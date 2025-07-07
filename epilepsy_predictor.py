import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')
ESR = pd.read_csv('Epileptic Seizure Recognition.csv (1).zip')
ESR = ESR.drop(columns = ESR.columns[0]) 
print(ESR.head())
cols = ESR.columns
tgt = ESR.y
tgt[tgt > 1] = 0
ESR.isnull().sum().sum()
print(ESR.describe())
Y = ESR.iloc[:,178].values
print(Y.shape)
Y[Y>1]=0
print(Y)
X = ESR.iloc[:,1:178].values
print(X.shape)

from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
new_input1 = ESR.iloc[6, :177].values.reshape(1, -1)
def predict_seizure(input_sample):
    sample_scaled = sc.transform(input_sample)
    prediction = clf.predict(sample_scaled)[0]
    return "Seizure" if prediction == 1 else "Non-Seizure"

scores = cross_val_score(clf, X, Y, cv=5)
print("Cross-validation accuracy scores:", scores)
print("Mean CV Accuracy:", scores.mean())

result = predict_seizure(new_input1)
print("Prediction:", result)

true_label = ESR.iloc[6, 178]  # Column 178 is the target (y)
print("Actual Label:", "Seizure" if true_label == 1 else "Non-Seizure")

