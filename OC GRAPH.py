from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
import pickle
from xgboost import XGBClassifier


DF = pd.read_csv("TRAININGDATA_new.csv")

udflist = list(DF.columns)

DF.infer_objects(copy=False)

DF1 = DF.interpolate() 

DF1['CA125'] = DF1['CA125'].astype(str).str.replace('>', '', regex=False)
DF1['CA125'] = pd.to_numeric(DF1['CA125'], errors='coerce')

DF2 = DF1[0:9]

X = np.asarray(DF2)
y = np.asarray(DF1['TYPE'])





