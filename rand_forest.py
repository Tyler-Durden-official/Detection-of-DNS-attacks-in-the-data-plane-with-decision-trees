#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("tokenised.csv")
x_total=data.iloc[:,:-1]
y_total=data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x_total,y_total,test_size=0.3)
model=RandomForestClassifier()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
