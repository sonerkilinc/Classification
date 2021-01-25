import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

df=pd.read_csv(r"C:\Users\user\Desktop\2020-2021 GÜZ DÖNEMİ DERS NOTLARI\yapay zeka\data.csv")

print(df.head())

print(df['target'])
#1.soru
sns.countplot(df['target'])
#2.soru
sns.countplot(x="target",hue="sex",data=df,palette="Blues")
#3.soru
sns.countplot(df['age'])
#4.soru
sns.distplot(df["age"],kde=True)

#5.soru
X=df.drop('target',axis=1)

y=df["target"]
print(y)

#6.soru
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1)

#7.soru
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(max_iter=120)
logmodel.fit(X_train,y_train)

print(logmodel.coef_)
pred=logmodel.predict(X_test)
print(pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(logmodel,X_test,y_test)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(logmodel,X_test,y_test,display_labels=["(0) Not target","(1) target"],cmap=plt.cm.magma)

from sklearn.metrics import classification_report
print(classification_report(y_test,pred))

cm=confusion_matrix(y_test,pred)
print(cm)

#accuracy
print((cm[0][0]+cm[1][1])/X_test.shape[0])

#Specifity
print(cm[1][1]/(cm[1][1]+cm[0][1]))

#sensivity
print(cm[0][0]/(cm[0][0]+cm[1][0]))

#Naive bayes çözümü
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
nv=GaussianNB()
nv.fit(X_train,y_train)

y_pred=nv.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(nv,X_test,y_test)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(nv,X_test,y_test,display_labels=["(0) Not target","(1) target"],cmap=plt.cm.magma)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#accuracy
print((cm[0][0]+cm[1][1])/X_test.shape[0])

#Sensivity
print(cm[0][0]/(cm[0][0]+cm[1][0]))

#Specifity
print(cm[1][1]/(cm[1][1]+cm[0][1]))

#knn sorusunun cevabı

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)

sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

import math
print(math.sqrt(len(X.values)))

classifier=KNeighborsClassifier(n_neighbors=18,p=2,metric='euclidean')
classifier.fit(X_train,y_train)

print(classifier.score(X_test,y_test))

y_pred=classifier.predict(X_test)
print(y_pred)

cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier,X_test,y_test)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier,X_test,y_test,display_labels=["(0) Not target","(1) target"],cmap=plt.cm.Blues)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#accuracy
print((cm[0][0]+cm[1][1])/X_test.shape[0])

#Sensivity
print(cm[0][0]/(cm[0][0]+cm[1][0]))

#Specifity
print(cm[1][1]/(cm[1][1]+cm[0][1]))

#karar ağacı sorusu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


X=df.iloc[:,0:13]
y=df.iloc[:,13]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.20)

sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred=dt.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(dt,X_test,y_test)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(dt,X_test,y_test,display_labels=["(0) Not target","(1) target"],cmap=plt.cm.Blues)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#accuracy
print((cm[0][0]+cm[1][1])/X_test.shape[0])

#Sensivity
print(cm[0][0]/(cm[0][0]+cm[1][0]))

#Specifity
print(cm[1][1]/(cm[1][1]+cm[0][1]))


#yapay sinir ağları
from sklearn.model_selection import train_test_split

X=df.iloc[:, 0:13]
y=df.iloc[:, 13]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

classifiers = Sequential()

classifiers.add(Dense(units=5,activation='relu',input_dim=13))
classifiers.add(Dense(units=3,activation='relu'))
classifiers.add(Dense(units=1,activation='sigmoid'))

classifiers.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

classifiers.fit(X_train, y_train, batch_size = 10, epochs = 100)

Y_pred = classifiers.predict(X_test)
Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred]
print(Y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
print(cm)


from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))

#accuracy
print("Accuracy:")
print((cm[0][0]+cm[1][1])/X_test.shape[0])

#Sensivity
print("Sensivity")
print(cm[0][0]/(cm[0][0]+cm[1][0]))

#Specifity
print("Specifity")
print(cm[1][1]/(cm[1][1]+cm[0][1]))



