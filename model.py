import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:\\Users\\SowmyaEppalpalli\\Downloads\\framingham.csv")

data.dropna()

dataset = data.reset_index(drop = True)

dataset = dataset.dropna(how = 'any')
df = dataset.reset_index(drop = True)

X=df.drop(['TenYearCHD'],axis=1)
y=df['TenYearCHD']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(X_train,y_train)


import pickle

pickle.dump(lg, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
