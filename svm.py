#Sample code for SVM
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
#Importing the dataset
df = pd.read_csv('Iris.csv')
df.head()
df.shape   
df.drop(columns=['Id'], inplace=True)

#Dividing dataset into features(X) and label(y)
X = df.drop(columns=['Species'])
y = df['Species']
#Splitting dataset to train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=45,stratify=y)
#If we specify random state as constant integer train data will be constant For every run otherwise
#Train data will be changed for every run and accuracy will differ

#Create the SVM model
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
# SVC with linear kernel

C = 1.0  # SVM regularization parameter

# LinearSVC (linear kernel)
svc =SVC(kernel='linear', C=C)
svc.fit(X_train, y_train)

# Evaluate the trainong model (Accuracy of training model)

Trainaccuracy = cross_val_score(svc,X_train,y_train,cv=5)
svc.fit(X_train,y_train)
print("Train model accuracy:", np.mean(Trainaccuracy))

# Evaluate the Testing accuracy
y_pred1 = svc.predict(X_test)
print ('Accuracy of linear svm:', accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
# SVC with RBF kernel
# Gamma indicates the ‘spread’ of the kernel  that is decision region.
#Gamma is low, the decision region is very broad. 
#When gamma is high, the ‘decision boundary is high, which creates islands of decision-boundaries around data points

rbf_svc =SVC(kernel='rbf', gamma=0.8, C=C)  
rbf_svc.fit(X_train, y_train)
y_pred2 = rbf_svc.predict(X_test)
print ('Accuracy of rbf kernel:', accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

# SVC with polynomial (degree 3) kernel
poly_svc =SVC(kernel='poly', degree=5, C=C)
poly_svc.fit(X, y)
y_pred3 = poly_svc.predict(X_test)
print ('Accuracy of polynomial kernel:', accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))

grid = {

    'C':[0.01,0.1,1,10],
    'kernel' : ["linear","poly","rbf"],
    'degree' : [1,3,5,7],
    'gamma' : [0.01,1]
}
svm =SVC()
svm_cv = GridSearchCV(svm, grid)
svm_cv.fit(X_train,y_train)

print("Best Parameters:",svm_cv.best_params_)

#print("Train Score:",svm_cv.best_score_)
print("Test Score:",svm_cv.score(X_test,y_test))
