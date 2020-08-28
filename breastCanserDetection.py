import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



data = pd.read_csv('BreastCancer_data.csv',header=0)
print(data.shape)



print(data.head())
print(list(data.columns))


print(data['diagnosis'].value_counts())


print(data.groupby('diagnosis').mean())
data.diagnosis.hist()

data.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
print(data.shape)


label = data.diagnosis
print(label.shape)


data.drop(["diagnosis"], axis = 1, inplace = True)
print("/n/n ",data.shape)


X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.3, random_state = 0)

lr = LogisticRegression()
lr.fit(X_train,y_train)

pred = lr.predict(X_test)
aucc = format(lr.score(X_test,y_test))
print("Accuracy of this model in test set is :",aucc)


con_mat = confusion_matrix(label, lr.predict(data))
print(con_mat)