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
        plt.text(x+0.2, y+0.005, str(round(y, 3)))
    plt.title("polynomial kernel")
    plt.xlabel("log(C)")
    plt.ylabel("Ein")
    plt.savefig("./polynomial.png")
    print("Completed 100%!")
    return 0

def main():
    # loading data
    print("Problem 14:")
    filename = ["./features.train","./features.test"]
    train_data = load_data(filename[0])
    test_data = load_data(filename[1])
    train_label = data_label(train_data,4)
    print("train_data size : ",train_data.shape)
    print("train_label size : ",train_label.shape)
    print("test_data size  : ",test_data.shape)

    # trian SVM
    print("--"*10)
    print("start training...")
    c = [ -5. , -3. , -1. , 1. , 3. ]
    E_in = []
    for i in range(5):
        print("completed : %d/5" %(i+1))
        clf = SVC(C = 10**c[i] , kernel = 'poly' , tol = 1e-4 ,coef0 = 1. , degree =2 , gamma=1)
        clf.fit(train_data[:,1:], train_label)
        tmp = 1. - clf.score(train_data[:,1:],train_label)
        print("E_in : ",tmp)
        E_in.append(tmp)

    plot_image(c,E_in)


# main()
main()
