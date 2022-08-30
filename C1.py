#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#reading the file
mc = pd.read_csv('Startups.csv',header = 0)
mc.head()
#Extracting Independent Values in x
x = mc[['R&D Spend','Administration','Marketing Spend']]

#Extracting Dependent Values in y
y = mc['Profit']
#plotting them before performing regression
plt.scatter(mc['Marketing Spend'], mc['Profit'])
model = linear_model.LinearRegression()
model.fit(x,y)

y_pred = model.predict(x)

#plotting after performing regression
plt.scatter(mc['Marketing Spend'], mc['Profit'])
plt.plot(x, y_pred, color='red')

#splitting the data,80% for training and 20% for testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#printing the training data used (Independent values)
print(x_train)
print("\n")

#printing the testing data used
print(x_test)
#printing the training data used (Dependent values)
print(y_train)
print("\n")
#printing the testing data used
print(y_test)

print("shape of original dataset :", mc.shape)
print("shape of input - training set", x_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", x_test.shape)
print("shape of output - testing set", y_test.shape)

#printing the R2 score
print ("Score = ",model.score(x, y))

#regression coefficient
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = [[165349.2], [162597.7], [153441.51], [144372.41], [142107.34]]
y = [[192261.83], [191792.06], [182901.99], [166187.94], [156991.12]]
model = LinearRegression()
model.fit(x,y)
plt.figure()
plt.title("Regression Coefficient (Linear Regression)")
plt.ylabel("Profit")
plt.xlabel("R & D Spend")
plt.plot(x,y)
plt.plot(x,model.predict(x),'-')
plt.axis([0,170000,0,195000])
plt.grid(True)
print (model.predict([[160000]])) # Predicted the average glucose level
plt.show()

#KNN
#reading the file
df = pd.read_csv("diabetes.csv")

#checking for null values
df.isnull().sum()

df.describe()
df.info()
scaler = StandardScaler()
#calling the function standardscaler
df.head()
scaler.fit(df.drop('Outcome', axis = 1))
#standardizing my class label
scaled_features = scaler.transform(df.drop('Outcome', axis = 1))
#saving the transformed in scaled_features
scaled_features
df[['BMI','Age','Pregnancies']].boxplot(grid = 'True')
df[['Glucose','BloodPressure','Insulin']].boxplot(grid = 'True')
sns.stripplot(data = df[['Glucose','BloodPressure','Insulin']])
sns.violinplot(data = df[['BMI','Age','Pregnancies']])
sns.heatmap(data = df)
sns.heatmap(data = df.corr(),square = True,annot = True)
import seaborn as sns
sns.pairplot(df , hue='Outcome')
#plotting my class label

x_train1 , x_test1 , y_train1 , y_test1 = train_test_split(scaled_features,df['Outcome'],test_size=0.40)
#dividing my dataset into testing data and training data into 60 40
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
classifier.fit(x_train1, y_train1)
#finding knn using k value as 10 and distance measure as Eucledian distance

y_pred1 = classifier.predict(x_test1)
#prediction after classification

from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(y_test1, y_pred1)
ac1 = accuracy_score(y_test1,y_pred1)
#finding the confusion matrix and accuracy matrix

#printing the confusion matrix
cm1

#printing the accuracy
ac1

#printing classifictaion report
from sklearn.metrics import classification_report
print(classification_report(y_test1,y_pred1))

#doing knn for k value = 20
scaled_features

x_train2 , x_test2 , y_train2 , y_test2 = train_test_split(df[['BMI',"Age"]],df['Outcome'],test_size=0.40)
#dividing my dataset into testing data and training data into 60 40

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 2)
classifier.fit(x_train2, y_train2)
#finding knn using k value as 10 and distance measure as Eucledian distance

y_pred2 = classifier.predict(x_test2)
#prediction after classification

from sklearn.metrics import confusion_matrix,accuracy_score
cm2 = confusion_matrix(y_test1, y_pred1)
ac2 = accuracy_score(y_test1,y_pred1)
#finding the confusion matrix and accuracy matrix

#printing the confusion matrix
cm2

#printing the accuracy
ac2

from sklearn.metrics import classification_report
print(classification_report(y_test2,y_pred2))
#printing classification report

#plotting graph between two accuracy when k value 

acc = []
from sklearn import metrics
for i in range(1,6):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(x_train1,y_train1)
    yhat = neigh.predict(x_test1)
    acc.append(metrics.accuracy_score(y_test1, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,6),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))
#running a for loop from 1 to 6 on minkowski distance measure

