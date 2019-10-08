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

    return out_data

class k_means():
    def __init__(self, k = 1):
        self.k = k
        self.x = None
        self.S = None
        self.dist = None
        self.group = None

    def fit(self, x):
        self.group = np.zeros(x.shape[0])
        init_select = np.random.choice(range(self.group.shape[0]), self.k)
        means = x[init_select]
        while True:
            self.dist = distance_matrix(x, means)
            data_tmp = []
            for each_point in self.dist:
                each_point_group = each_point.argmin()
                data_tmp.append(each_point_group)
            # create new group
            new_group = np.array(data_tmp).reshape(-1)
            if np.mean(new_group != self.group) == 0.:
                break
            self.group = new_group
            # create new mean
            mean_tmp = []
            for i in range(self.k):
                mean_tmp.append(x[self.group == i].mean(axis = 0))
            means = np.array(mean_tmp)
            means = np.nan_to_num(means)
        #print(self.group.shape)

    def Error(self):
        dist_sum = 0.
        N = self.dist.shape[0]
        for i in range(self.dist.shape[0]):
            belong_group = int(self.group[i])

            same_group_dist = self.dist[i][belong_group]
            dist_sum += same_group_dist**2.
        return dist_sum/N

def main():
    parameter_k = [2,4,6,8,10]


    filename = ["./data/hw4_nolabel_train.dat"]
    train_data = load_file(filename[0])
    print('train_data shape:', train_data.shape)

    print("Problem 15:")
    E_in = []
    for k in parameter_k:
        e_in = []
        clf = k_means(k = k)
        for iter in range(500):
            clf.fit(train_data)
            e_in.append(clf.Error())
        E_in.append(e_in)

    E_avg = []
    E_var = []
    for i in range(len(E_in)):
        e_in = E_in[i]
        e_in = np.array(e_in)
        mean_tmp = e_in.mean()
        var_tmp = e_in.var()
        E_avg.append(mean_tmp)
        E_var.append(var_tmp)

    plt.figure(figsize = (8,6))
    plt.ylabel('E_avg(blue)')
    plt.xlabel('k')
    plt.plot(parameter_k,E_avg,'-o')
    plt.savefig('E_avg(k_means).png')
    print("done!")

    print("\nProblem 16:")
    plt.figure(figsize = (8,6))
    plt.ylabel('E_var(blue)')
    plt.xlabel('k')
    plt.plot(parameter_k,E_var,'-o')
    plt.savefig('E_var(k_means).png')
    print("done!")



if __name__ == "__main__":
    main()
