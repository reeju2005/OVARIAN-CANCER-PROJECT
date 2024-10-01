from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import pickle
from xgboost import XGBClassifier


DF = pd.read_csv("TRAININGDATA_new.csv")

udflist = list(DF.columns)

DF.infer_objects(copy=False)

DF1 = DF.interpolate() 

DF1['CA125'] = DF1['CA125'].astype(str).str.replace('>', '', regex=False)
DF1['CA125'] = pd.to_numeric(DF1['CA125'], errors='coerce')


X = DF1.drop('TYPE', axis=1)
y = DF1[['TYPE']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)

model = XGBClassifier(tree_method='approx', max_bin=120, n_estimators=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'F1 Score: {f1:.3f}')
print(f'Recall: {recall:.3f}')

pickle.dump(model, open('model.pkl', 'wb'))