import numpy as np
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import random
import os

# data processing
def load_file(filename):
    f = open(filename,'r')
    train_data = []
    for line in f:
        tmp_list = []
        line = line.split()
        for i in range(len(line)):
            tmp_list.append(float(line[i]))
        tmp_list = np.array(tmp_list)
        train_data.append(tmp_list)
    train_data = np.array(train_data)
    return train_data

def data_loader(data_t):
    data_m = data_t.copy()
    data = data_m[:,:-1]
    label = data_m[:,-1]
    return data,label

# Adaboost
def s_classify(X, i, theta , s_kind):
    # s : +1,-1  , i : dimension of the data
    pred = np.ones(X.shape[0])
    if s_kind == 1:
        mask = (X[:, i] < theta)
        pred[mask] = -1
    else:
        mask = (X[:, i] >= theta)
        pred[mask] = -1
    return pred

def decisionstump(X_, Y_, u_):
    N, dimension = X_.shape
    minErr = np.inf
    min01Err = np.inf
    bestStump = {}
    for j in range(dimension):
        idx = np.argsort(X_[:, j])
        un_idx = np.argsort(idx)
        X = X_[idx].copy()
        Y = Y_[idx].copy()
        u = u_[idx].copy()
        tmp_X = np.hstack((-np.inf,X[:,j]))
        tmp_X = np.delete(tmp_X,-1)
        theta = ( tmp_X + X[:,j] ) / 2.
        for i,theta in enumerate(theta):
            for s in [1 , -1]:
                pred = s_classify(X, j, theta, s)
                errLabel = np.zeros(N)
                errLabel[pred != Y] = 1.
                weightedErr = np.dot(u,errLabel)/N
                if minErr > weightedErr:
                    min01Err = np.sum(errLabel)/N
                    minErr = weightedErr
                    bestClass = pred[un_idx]
                    epsilon = (minErr * N)/np.sum(u)
                    bestStump['dimension'] = j
                    bestStump['theta'] = theta
                    bestStump['s'] = s
    return bestStump, minErr, bestClass , epsilon , min01Err

def main():
    print("Problem 13,14,15,16:")
    filename = ["./data/hw2_adaboost_train.dat","./data/hw2_adaboost_test.dat"]
    train = load_file(filename[0])
    test = load_file(filename[1])
    data_train,label_train = data_loader(train)
    data_test,label_test = data_loader(test)
    print("training data size :",data_train.shape)
    print("testing data size :",data_test.shape)

    dimension = data_train.shape[1]
    numberofdata = data_train.shape[0]
    T = 300
    sorted_idx = []
    u = np.array([ 1. / numberofdata ] * numberofdata) # shape : (numberofdata,1)
    alpha = np.zeros(T)
    U = []
    Ein_g = []
    Ein_G = []
    Eout_G = []
    Epsilon = []
    G = []

    for iter in range(T):
        if iter % 50 == 0:
            print('iteration = %d/300' % (iter))
        # Decision Stump

        bestStump, minErr, bestPred, epsilon ,min01Err= decisionstump(data_train,label_train,u)

        g = [bestStump['dimension'], bestStump['theta'], bestStump['s']]
        # predict by small gt
        # print("epsilon",epsilon)
        scale = np.sqrt( ( 1. - epsilon) / epsilon )
        # update u
        mask_incorrect = (bestPred != label_train)
        mask_correct = (bestPred == label_train)
        u[mask_incorrect] *= scale
        u[mask_correct] /= scale
        U.append( u.sum() )

        alpha[iter] = np.log( scale )
        Ein_g.append( min01Err )
        Epsilon.append( epsilon )
        G.append(g)

        # predict by big Gt
        result = []
        for x in data_train:
            # g(d, theta, s)
            predict = 0
            for (d, theta, s), a in zip(G, alpha):
                predict += a * s * np.sign(x[d] - theta)
            result.append( np.sign(predict) )
        pred = np.array(result)

        err_G = np.sum( pred != label_train ) / numberofdata
        Ein_G.append(err_G)

        # predict test
        result = []
        for x in data_test:
            # g(d, theta, s)
            predict = 0
            for (d, theta, s), a in zip(G, alpha):
                predict += a * s * np.sign(x[d] - theta)
            result.append( np.sign(predict) )
        pred = np.array(result)

        err_G = np.sum( pred != label_test ) / data_test.shape[0]
        Eout_G.append(err_G)

    # problem 13
    print("HI!Because of processing training & testing at the same time,")
    print("it will cost some time.")
    print('\nProblem 13')
    print('Ein(gT):', Ein_g[-1], ', alpha_T:', alpha[-1])
    # print("Ein gt : ",Ein_g)
    plt.figure()
    plt.plot(Ein_g, 'b')
    plt.xlabel('t')
    plt.ylabel('0/1 error')
    plt.title('t vs. Ein(gt)')
    plt.show()

    # problem 14
    print('\nProblem 14')
    print('Ein(GT):', Ein_G[-1])
    plt.figure()
    plt.plot(Ein_G, 'b')
    plt.xlabel('t')
    plt.ylabel('0/1 error')
    plt.title('t vs. Ein(Gt)')
    plt.show()

    # problem 15
    print('\nProblem 15')
    print('U(T):', U[-1])
    plt.figure()
    plt.plot(U, 'b')
    plt.xlabel('t')
    plt.ylabel('Ut')
    plt.title('t vs. Ut')
    plt.show()

    # problem 16
    print('\nProblem 16')
    print('Eout(GT):', Eout_G[-1])
    plt.figure()
    plt.plot(Eout_G, 'b')
    plt.xlabel('t')
    plt.ylabel('0/1 error')
    plt.title('t vs. Eout(Gt)')
    plt.show()

if __name__ == "__main__":
    main()
