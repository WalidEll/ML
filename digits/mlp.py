import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


scaler = StandardScaler()
# Fit only to the training data

digits = load_digits()

X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target)

scaler.fit(X_train)
scaler.fit(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,500,50),activation='relu')
mlp.fit(X_train,Y_train)
predictions = mlp.predict(X_test)

print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
