import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#importing libraries
df = pd.read_csv("/content/indian_airquality2.csv")
#importing dataset

df.head()
#printing the head of the dataset
df["pollutant_avg"].mean()
#finding mean of the pollutant_avg column
def categorise(row):  
    if row['pollutant_avg'] > 54:
        return 1
    else :
        return 0

df['lable'] = df.apply(lambda row: categorise(row), axis=1)
#making a new column names as lable which will be my class lable
df.head()
from sklearn.preprocessing import StandardScaler
#importing sklearn library
scaler = StandardScaler()
#calling the function standardscaler
df2 = df.drop(['id','last_update','station','city','country','pollutant_min','pollutant_max','pollutant_avg'], axis=1)
#dropping the column which won't be used
df2
df2['state'].unique()
df2['pollutant_id'].unique()
df2['pollutant_id'].replace(['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'],
                        [0, 1,2,3,4,5,6], inplace=True)
df2['state'].replace(['Andhra_Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
       'Delhi', 'Gujarat', 'Haryana', 'Jammu_&_Kashmir', 'Jharkhand',
       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry',
       'Punjab', 'Rajasthan', 'TamilNadu', 'Telangana', 'Tripura',
       'Uttar_Pradesh', 'West_Bengal'],
                        [0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], inplace=True)
X = df2.iloc[:, :2].values
y = df2.iloc[:, -1].values
X
y
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
drugTree.fit(X_trainset,y_trainset)
predTree = drugTree.predict(X_testset)
print (predTree [0:3])
print (y_testset [0:3])
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

from six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier(criterion = 'entropy')
dtree.fit(X_trainset,y_trainset)
predictions= dtree.predict(X_testset)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_testset,predictions))
print('\n')
print(classification_report(y_testset,predictions))
tree.plot_tree(dtree)
