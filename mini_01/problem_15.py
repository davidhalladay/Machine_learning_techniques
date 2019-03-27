import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


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

def plot_image(c,w_norm):
    plt.figure(figsize = (10,8))
    plt.plot(c,w_norm, '--o')
    for x, y in zip(c, w_norm):
        plt.text(x+0.2, y+0.02, str(round(y, 3)))
    plt.title("Gaussian kernel")
    plt.xlabel("log(C)")
    plt.ylabel("distance")
    plt.savefig("./Gaussian.png")
    print("Completed 100%!")
    return 0

def main():
    # loading data
    print("Problem 15:")
    filename = ["./features.train","./features.test"]
    train_data = load_data(filename[0])
    test_data = load_data(filename[1])
    train_label = data_label(train_data,0)
    print("train_data size : ",train_data.shape)
    print("train_label size : ",train_label.shape)
    print("test_data size  : ",test_data.shape)

    # trian SVM
    print("--"*10)
    print("start training...")
    c = [ -2. , -1. , 0. , 1. , 2. ]
    dist = []
    for n in range(5):
        print("completed : %d/5" %(n+1))
        clf = SVC(C = 10**c[n] , kernel = 'rbf' , tol = 1e-5 ,gamma = 80)
        clf.fit(train_data[:,1:], train_label)

        print("finding parameters w")
        #find norm of w
        w = 0.
        for i ,ayi in zip(clf.support_,clf.dual_coef_[0]):
            for j , ayj in zip(clf.support_,clf.dual_coef_[0]):
                w += ayi * ayj * np.exp(-80*np.linalg.norm(train_data[j,1:]-train_data[i,1:])**2.)
        w = np.sqrt(w)

        print("finding distance")
        #dist.
        tmp = 1./np.abs(w)

        print("dist : ",tmp)
        dist.append(tmp)

    plot_image(c,dist)


# main()
main()
