from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
X=[2,2,7.5,4,3,3,0.5,5,6,4]
X=np.reshape(X, (-1,2)) 
print(X)
Z=linkage(X,'ward')
dendrogram(Z)
plt.savefig("cha.png")
plt.show()
