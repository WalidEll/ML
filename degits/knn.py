import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
digits = load_digits()

X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target)


knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,Y_train)

y_pred=knn.predict(X_test)
print(accuracy_score(Y_test,y_pred))


print(accuracy_score(Y_test,y_pred))
print(y_pred[0])
print(Y_test[0])

plt.gray()
plt.matshow(X_test[0].reshape((8,8)))
plt.title("Predected:"+str(y_pred[0])+" True:"+str(Y_test[0]))
plt.show()
