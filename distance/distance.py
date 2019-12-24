import numpy as np
import pandas as pd
points = np.random.rand(5,2)
names = ['A','B','C','D','E']
distances=np.empty((5,5))
for i in range(5):
    for j in range(5):
        distances[i][j] = np.linalg.norm(points[i]-points[j])
distDF = pd.DataFrame(distances, columns = names, index=names)
print(distDF)
