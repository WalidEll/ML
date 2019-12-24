from sklearn.neighbors import KNeighborsClassifier

X = [[3,3], [4,3.8], [5,3], [6,3.5],[3,4],[4,4.2],[4,6],[6,5]]
y = [0, 0, 0, 0,1,1,1,1]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print(knn.predict([[5.,3.5]]))

print(knn.predict_proba([[5.,3.5]]))
