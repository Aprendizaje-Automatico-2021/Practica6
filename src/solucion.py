from sklearn.svm import SVC
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

#------------LOAD-DATA------------#
def loadData1():
	data = loadmat('./src/assets/ex6data1.mat')
	X = data['X']
	y = data['y']

	return X, y

def loadData2():
    data = loadmat('./src/assets/ex6data2.mat')
    X = data['X']
    y = data['y']

    return X, y
    
def loadData3():
    data = loadmat('./src/assets/ex6data3.mat')
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']

    return X, y, Xval, yval

#--------------OTROS---------------#
def selectParameters(X, y, Xval, yval, initialValue, iter):
    bestScore = 0
    bestSvm = 0
    
    for i in range(iter):
        reg = initialValue * 3**i
        for j in range(iter):
            sigma = initialValue * 3**j
            svm = SVC(kernel='rbf', C=reg, gamma= 1 / (2 * sigma**2))
            svm.fit(X,y.ravel())
            accuracy = accuracy_score(yval, svm.predict(Xval))
            if(accuracy > bestScore):
                bestSvm = svm
                bestScore = accuracy

    return bestSvm

#--------------DRAW---------------#
def drawKernel(X, y, svm, zoom=True):
    fig = plt.figure()
    # Genera 100 valores separados por una cte desde min hasta max
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    # Genera la malla entre x1 y x2
    x1, x2 = np.meshgrid(x1, x2)
    # Predicción de los valores y a partir del svm
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='black')
    plt.scatter(X[neg, 0], X[neg, 1], c='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    # Pone los límites X e Y de la gráfica
    if zoom:
        offsetX = 0.25
        offsetY = 0.1
        plt.axis([X[:, 0].min() - offsetX, X[:, 0].max() + offsetX,
                X[:, 1].min() - offsetY, X[:, 1].max() + offsetY])
    
    plt.show()

#--------------PARTES---------------#
def part1():
    X, y = loadData1()
    reg = 1.0   # Término de regularización
    #-------KERNEL-LINEAL-------#
    # Genera el SVM
    svm = SVC(kernel='linear', C=reg)
    # Entrenamiento del SVM
    svm.fit(X, y.ravel())
    # Dibujo de la gráfica con el svm entrenado
    drawKernel(X, y, svm)
    # Prueba de la nueva gráfica con distinto valor para reg
    reg = 100.0
    svm = SVC(kernel='linear', C=reg)
    svm.fit(X, y.ravel())
    drawKernel(X, y, svm, False)
    #-------KERNEL-GAUSSIANO-------#
    X, y = loadData2()
    reg = 1.0
    sigma = 0.1
    svm = SVC(kernel='rbf', C=reg, gamma = 1 / (2 * sigma**2))
    svm.fit(X, y.ravel())
    drawKernel(X, y, svm, False)
    #-------ELECCION-C-SIGMA-------#
    X, y, Xval, yval = loadData3()
    initialValue = 0.01
    svm = selectParameters(X, y, Xval, yval, initialValue, 7)
    drawKernel(X, y, svm)

def main():
    # Parte 1
    part1()

    return 0

#------------------------------------#
main()