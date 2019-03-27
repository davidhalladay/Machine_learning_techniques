import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import random
from sklearn.model_selection import train_test_split

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
        plt.text(x+0.2, y+0.1, str(round(y, 3)))
    plt.title("Gaussian kernel")
    plt.xlabel("log(C)")
    plt.ylabel("distance")
    plt.savefig("./Gaussian.png")
    print("Completed 100%!")
    return 0

def main():
    # loading data
    print("Problem 16:")
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
    r = [-2. ,-1.,0.,1.,2.]
    r_count = [0,0,0,0,0]
    for count in range(100):
        print("completed %d/100" %count)
        r_Eva = []
        train_data_t, valid_data_t, train_label_t, valid_label_t = train_test_split(train_data[:,1:], train_label,test_size = 1000)
        #print(valid_data_t)
        #print(valid_label_t)
        for i in range(5):
            clf = SVC(C = 0.1 , kernel = 'rbf' , tol = 1e-3 ,gamma = 10.**r[i])
            clf.fit(train_data_t, train_label_t)
            E_va = 1. - clf.score(valid_data_t,valid_label_t)
            r_Eva.append(E_va)
        min_t = 0
        for i in range(1,len(r_Eva)):
            if (r_Eva[i] < r_Eva[min_t]): min_t = i
        r_count[min_t] = r_count[min_t] + 1
        print("select : ",r_Eva[min_t])
    print(r_count)

    plt.figure(figsize=(10,8))
    r_l = [ "0.01" ,"0.1","0","10","100"]
    y_pos = np.arange(len(r))
    plt.bar(y_pos, r_count, align='center', alpha=0.5)
    plt.xticks(y_pos, r_l)
    plt.ylabel('count')
    plt.title('r_count hist')
    plt.savefig('./hist.png')



# main()
main()
