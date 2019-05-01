import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import random

# loading data
# i/p: string(filename) ,o/p: np.array(data)
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
    data = data_t.copy()
    data = np.hstack([np.ones((500, 1)), data])
    train = data[:400]
    test = data[400:]
    train_data = train[:,:-1]
    train_label = train[:,-1]
    test_data = test[:,:-1]
    test_label = test[:,-1]
    return train_data,train_label,test_data,test_label

def main():
    print("Problem 9,10:")
    filename = ["./data/hw2_lssvm_all.dat"]
    data = load_file(filename[0])
    train_data,train_label,test_data,test_label = data_loader(data)

    for lambda_ in [0.05, 0.5, 5, 50, 500]:
        # copy : https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
        clf = KernelRidge(alpha = lambda_ ,kernel='linear')
        clf.fit(train_data, train_label)
        pred_train = clf.predict(train_data)
        pred_test = clf.predict(test_data)
        # predict
        count = 0
        for i in range(pred_train.shape[0]):
            if np.sign(pred_train[i]) != train_label[i]:
                count += 1
        Ein = count/float(pred_train.shape[0])
        count = 0
        for i in range(pred_test.shape[0]):
            if np.sign(pred_test[i]) != test_label[i]:
                count += 1
        Eout = count/float(pred_test.shape[0])

        print("lambda = %f ,Ein = %.4f ,Eout = %.4f" %(lambda_, Ein, Eout))

    print("Problem 11,12:")
    for lambda_ in [0.05, 0.5, 5, 50, 500]:
        ein = np.zeros(400)
        eout = np.zeros(100)
        for i in range(250):
            clf = KernelRidge(alpha=lambda_,kernel='linear')
            idx = np.random.choice(400, size=400)
            clf.fit(train_data[idx], train_label[idx])
            ein += np.sign(clf.predict(train_data))
            eout += np.sign(clf.predict(test_data))
        # predict
        count = 0
        for i in range(np.sign(ein).shape[0]):
            if np.sign(np.sign(ein)[i]) != train_label[i]:
                count += 1
        Ein = count/float(np.sign(ein).shape[0])
        count = 0
        for i in range(np.sign(eout).shape[0]):
            if np.sign(np.sign(eout)[i]) != test_label[i]:
                count += 1
        Eout = count/float(np.sign(eout).shape[0])

        print("lambda = %f ,Ein = %.4f ,Eout = %.4f" %(lambda_, Ein, Eout))

if __name__ == "__main__":
    main()
