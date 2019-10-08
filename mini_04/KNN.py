import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance_matrix

# distance reference
# https://blog.csdn.net/pipisorry/article/details/48814183
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance_matrix.html

# loading data
# i/p: string(filename) ,o/p: np.array(data)
def load_file(filename):
    f = open(filename,'r')
    data = []
    for line in f:
        tmp_list = []
        line = line.split()
        for i in range(len(line)):
            tmp_list.append(float(line[i]))
        tmp_list = np.array(tmp_list)
        data.append(tmp_list)
    out_data = np.array(data)

    return out_data[:,:-1] , out_data[:,-1]

class my_KNN():
    def __init__(self, k = 1):
        self.k = k
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        dist_x = distance_matrix(x, self.x)
        #print(dist_x.shape)
        arg_dist_x = dist_x.argsort(axis = 1)
        nearest_tmp = []
        for idx in arg_dist_x:
            nearest_tmp.append(np.array([self.y[idx[:self.k]]]))
        nearest_tmp = np.array(nearest_tmp)
        num = nearest_tmp.shape[0]
        nearest_tmp = nearest_tmp.reshape((num,-1))
        #print(nearest_tmp.shape)
        #print(nearest_tmp)
        return np.sign(np.sum(nearest_tmp,axis = 1))

class KNN():
    def __init__(self, k = 1,gamma = 1):
        self.k = k
        self.x = None
        self.y = None
        self.gamma = gamma

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        dist_x = distance_matrix(x, self.x)
        #print(dist_x.shape)
        dist_x = -1. * (dist_x**2.) * self.gamma
        dist_x = np.exp(dist_x)
        num_x = dist_x.shape[0]
        #print(dist_x.shape)
        #print(self.y.shape)
        tmp = []
        for idx in range(num_x):
            tmp.append(np.dot(self.y,dist_x[idx]))
        output = np.array(tmp)
        #print(tmp.shape)
        #num = tmp.shape[0]
        #output = tmp.reshape((num,-1))
        #print(output.shape)
        return np.sign(output)

def main():
    parameter_k = [1,3,5,7,9]
    parameter_gamma = [0.001, 0.1, 1., 10., 100.]
    parameter_log_gamma = [-3, -1, 0, 1., 2.]

    filename = ["./data/hw4_train.dat","./data/hw4_test.dat"]
    train_data ,train_label = load_file(filename[0])
    test_data ,test_label = load_file(filename[1])
    print('train_data shape:', train_data.shape)
    print('test_data shape:', test_data.shape)

    print("Problem 11:")
    E_in = []
    for k in parameter_k:
        clf = my_KNN(k = k)
        clf.fit(train_data, train_label)
        pred = clf.predict(train_data)
        e_in = np.mean(pred != train_label)
        E_in.append(e_in)

    plt.figure(figsize = (8,6))
    plt.ylabel('Ein(blue)')
    plt.xlabel('k')
    plt.plot(parameter_k,E_in,'-o')
    plt.savefig('E_in(my_KNN).png')
    print("done!")

    print("\nProblem 12:")
    E_out = []
    for k in parameter_k:
        clf = my_KNN(k = k)
        clf.fit(train_data, train_label)
        pred = clf.predict(test_data)
        e_out = np.mean(pred != test_label)
        E_out.append(e_out)

    plt.figure(figsize = (8,6))
    plt.ylabel('Eout(blue)')
    plt.xlabel('k')
    plt.plot(parameter_k,E_out,'-o')
    plt.savefig('E_out(my_KNN).png')
    print("done!")

    print("\nProblem 13:")
    E_in = []
    for gamma in parameter_gamma:
        clf = KNN(gamma = gamma)
        clf.fit(train_data, train_label)
        pred = clf.predict(train_data)
        e_in = np.mean(pred != train_label)
        E_in.append(e_in)

    plt.figure(figsize = (8,6))
    plt.ylabel('E_in(blue)')
    plt.xlabel('log(gamma)')
    plt.plot(parameter_log_gamma,E_in,'-o')
    plt.savefig('E_in(KNN).png')
    print("done!")

    print("\nProblem 14:")
    E_out = []
    for gamma in parameter_gamma:
        clf = KNN(gamma = gamma)
        clf.fit(train_data, train_label)
        pred = clf.predict(test_data)
        e_out = np.mean(pred != test_label)
        E_out.append(e_out)

    plt.figure(figsize = (8,6))
    plt.ylabel('E_out(blue)')
    plt.xlabel('log(gamma)')
    plt.plot(parameter_log_gamma,E_out,'-o')
    plt.savefig('E_out(KNN).png')
    print("done!")
    

if __name__ == "__main__":
    main()
