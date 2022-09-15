import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#importing libraries

df = pd.read_csv("/content/indian_airquality2.csv")
#importing dataset

df.head()
#printing the head of the dataset
from sklearn.preprocessing import StandardScaler
#importing sklearn library
scaler = StandardScaler()
#calling the function standardscaler
df2 = df[['state','pollutant_avg']]
#dropping the column which won't be used
df2 = df2.head(250)
df2['state'].unique()
df2['state'].replace(['Andhra_Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
       'Delhi'], [0,1,2,3,4,5], inplace=True)
df2.isnull().sum()
df2 = df2.dropna()

df2.head()
plt.scatter(df2.pollutant_avg,df2.state,marker='+',color='red')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = df2.iloc[:, :1].values
Y = df2.iloc[:, -1].values
X_Train, X_Test, y_Train, y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
X_Test
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_Train, y_Train)
X_Test

y_predicted = model.predict(X_Test)
model.predict_proba(X_Test)
model.score(X_Test,y_Test)
confusion_matrix(y_Test, y_predicted)
print(classification_report(y_Test, y_predicted))
