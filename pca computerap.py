import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
from sklearn import decomposition
 
df=pd.read_csv('iris.csv')
X=pd.get_dummies(df.loc[:, ['sepal_length', 'sepal_width','petal_length','petal_width']])
y=df.loc[:, 'species']
 
print("Original:\n %s" % X.head())
 
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
X = pd.DataFrame(X,columns=["1st PC","2nd PC","3rd PC"])
print("Transformed:\n %s" % X.head())
 
colorDict = { "setosa": "r", "virginica": "g","versicolor": "b" }
def applyColor(idx):
    return colorDict[idx]
 
X["Species"] = y
X["Color"] = X.Species.apply(applyColor)
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X["1st PC"], X["2nd PC"], X["3rd PC"], c=X["Color"], s=60)
ax.view_init(30, 185)
plt.show()
