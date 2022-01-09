from sklearn.svm import SVC
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from process_email import *

import matplotlib.pyplot as plt
import numpy as np
import codecs
import get_vocab_dict as getDict
ASSETS_PATH = './src/assets/'

#------------LOAD-DATA------------#
def loadData1():
	data = loadmat(ASSETS_PATH + 'ex6data1.mat')
	X = data['X']
	y = data['y']

	return X, y

def loadData2():
    data = loadmat(ASSETS_PATH + 'ex6data2.mat')
    X = data['X']
    y = data['y']

    return X, y
    
def loadData3():
    data = loadmat(ASSETS_PATH + 'ex6data3.mat')
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']

    return X, y, Xval, yval

def readEmails(path, vocab):
    email_contents = open(path, 'r', encoding='utf-8', errors='ignore').read()
    email = email2TokenList(email_contents)

    mailVector = np.zeros([len(vocab)])

    # A partir de cada palabara en el email se busca 
    # si dicha palabra está contenida en vocab y 
    # se indica con un 1.0 si es True
    for word in email:  
        if word in vocab:
            i = vocab[word] - 1
            mailVector[i] = 1

    return mailVector

def loadSpamData(path, row, vocab):
    spam = np.zeros([row, len(vocab)])
    for i in range(1, row + 1):
        textPath = path + str(i).zfill(4) + '.txt'
        spam[i - 1] = readEmails(textPath, vocab)
    
    return spam

#--------------OTROS---------------#
def selectParameters(X, y, Xval, yval, initialValue, iter):
    bestScore = 0
    bestSvm = 0

    reg = initialValue
    sigma = initialValue
    
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

    return bestSvm, reg, sigma, bestScore

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

def part2():
    numSpam = 500
    numEasyHam = 2551
    numHardHam = 250
    vocab = getDict.getVocabDict()

    print("Leyendo correo...")

    # Lectura de spam
    path = ASSETS_PATH + 'spam/'
    spamX = loadSpamData(path, numSpam, vocab)
    spamY = np.ones([numSpam])

    # Lectura EasyHam
    path = ASSETS_PATH + 'easy_Ham/'
    easyHamX = loadSpamData(path, numEasyHam, vocab)
    easyHamY = np.zeros([numEasyHam])

     # Lectura HardHam
    path = ASSETS_PATH + 'hard_ham/'
    hardHamX = loadSpamData(path, numHardHam, vocab)
    hardHamY = np.zeros([numHardHam])

    print("Fragmentando los datos de entrenamiento, validación y testing...")
    print(f"Seleccionando el 60% de datos para entrenamiento desde 0 hasta 59...")
    X = np.vstack((spamX[:int(0.6 * np.shape(spamX)[0])],
                easyHamX[:int(0.6 * np.shape(easyHamX)[0])],
                hardHamX[:int(0.6 * np.shape(hardHamX)[0])]))
    
    y = np.hstack((spamY[:int(0.6 * np.shape(spamY)[0])],
                easyHamY[:int(0.6 * np.shape(easyHamY)[0])],
                hardHamY[:int(0.6 * np.shape(hardHamY)[0])]))

    print(f"Seleccionando el 20% de datos para entrenamiento desde 60 hasta 79...")
    Xtest = np.vstack((spamX[int(0.6 * np.shape(spamX)[0]):int(0.8 * np.shape(spamX)[0])],
                    easyHamX[int(0.6 *np.shape(easyHamX)[0]):int(0.8 * np.shape(easyHamX)[0])],
                    hardHamX[int(0.6 * np.shape(hardHamX)[0]):int(0.8 * np.shape(hardHamX)[0])]))
    
    ytest = np.hstack((spamY[int(0.6 * np.shape(spamY)[0]):int(0.8 * np.shape(spamY)[0])],
                    easyHamY[int(0.6 * np.shape(easyHamY)[0]):int(0.8 * np.shape(easyHamY)[0])],
                    hardHamY[int(0.6 * np.shape(hardHamY)[0]):int(0.8 * np.shape(hardHamY)[0])]))
    

    print(f"Seleccionando el 20% de datos para entrenamiento desde 80 hasta 99...")
    Xval = np.vstack((spamX[int(0.8*np.shape(spamX)[0]):],
                        easyHamX[int(0.8 * np.shape(easyHamX)[0]):],
                        hardHamX[int(0.8 * np.shape(hardHamX)[0]):]))

    yval = np.hstack((spamY[int(0.8 * np.shape(spamY)[0]):],
                        easyHamY[int(0.8 * np.shape(easyHamY)[0]):],
                        hardHamY[int(0.8 * np.shape(hardHamY)[0]):]))

    print("Entrenando sistema de detección de spam")
    initialValue = 0.01

    svm, reg, sigma, bestScore = selectParameters(X, y, Xval, yval, initialValue, 8)
    testScore = svm.score(Xtest, ytest)
    print(f"Mejor C: {reg}")
    print(f"Mejor sigma: {sigma}")
    print(f"Error: {1 - bestScore}")
    print(f"Precisión: {testScore * 100}%")
    print("Success")

def main():
    # Parte 1
    #part1()

    # Parte 2
    part2()
    return 0

#------------------------------------#
main()