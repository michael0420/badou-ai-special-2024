import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn import datasets


#load iris data
iris_data = datasets.load_iris()
#print(iris_data.target[:5])
x,y=iris_data, iris_data.target#特征判断的类别

pca=dp.PCA(n_components=2)

new_iris_data=pca.fit_transform(x)

red_x,red_y = [],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]

for i in range(len(new_iris_data)):
    if y[i]==0:
        red_x.append(new_iris_data[i][0])
        red_y.append(new_iris_data[i][1])
    elif y[i]==1:
        blue_x.append(new_iris_data[i][0])
        blue_y.append(new_iris_data[i][1])
    else:
        green_x.append(new_iris_data[i][0])
        green_y.append(new_iris_data[i][1])

plt.scatter(red_x,red_y,c='r',marker='x',label='setosa')
plt.scatter(blue_x,blue_y,c='b',marker='D',label='Versicolour')
if green_x and green_y:
    plt.scatter(green_x, green_y, c='g', marker='.', label='Virginica')
plt.legend()
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('PCA of Iris Dataset')
plt.show()


