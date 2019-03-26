import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import my_lib

# loading data
# i/p: string(filename) ,o/p: np.array(data)
def load_data(filename):
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

# construct lebal
# i/p: np.array(data),int(classfy_num) ,o/p: np.array(label)
def data_label(data,classfy_num):
    label = []
    classfy_num = float(classfy_num)
    for i in data:
        if (i[0] == classfy_num):
            label.append(1.)
        else :
            label.append(0.)
    return np.array(label)

def main():
    # loading data
    filename = ["./features.train","./features.test"]
    train_data = load_data(filename[0])
    test_data = load_data(filename[1])
    train_label = data_label(train_data,2)
    print("train_data size : ",train_data.shape)
    print("train_label size : ",train_label.shape)
    print("test_data size  : ",test_data.shape)

    # trian SVM
    c = [ -5. , -3. , -1. , 1. , 3. ]
    clf = SVC(C = 10**c[0] , kernel = 'linear')
    clf.fit(train_data, train_label)

    # calculate np.abs(w)
    w = clf.coef_

    alpha = np.abs(clf.dual_coef_.reshape(-1))
    w_d = np.array([0.,0.,0.])
    for n in range(clf.support_vectors_.shape[0]):
        tmp = clf.dual_coef_.reshape(-1)[n] * clf.support_vectors_[n]
        w_d += tmp
    print(w_d)





# main()
main()
