from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import random
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import time

output_file_name = "RandomForestRegressor_01.csv"
# total: 47500
train_size = 30000
valid_size = 17500
MSD_feature_size = 10
VAC_feature_size = 0
T = 3

def load_file(file_path):
    data = np.load(file_path)['arr_0']
    return data

def write_file(data,filename):
    data = np.array(data)
    output_file_path = os.path.join("./submission",filename)
    numofdata = data.shape[1]
    numoffeature = data.shape[0]
    file = open(output_file_path,"w")

    for row in range(numofdata):
        text = data[:,row]
        file.write("%f,%f,%f\n"%(text[0],text[1],text[2]))

    return True

def dataloader(X,Y):
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_valid = X[train_size:(train_size+valid_size)]
    Y_valid = Y[train_size:(train_size+valid_size)]

    Y_train_frag = [Y_train[:,0],Y_train[:,1],Y_train[:,2]]
    Y_valid_frag = [Y_valid[:,0],Y_valid[:,1],Y_valid[:,2]]
    Y_train_frag = np.array(Y_train_frag)
    Y_valid_frag = np.array(Y_valid_frag)
    return X_train,Y_train_frag,X_valid,Y_valid_frag

def WMAE(pred_Y,gt_Y):
    # input : (3 ,num of data)
    w = [300,1,200]
    output = 0.
    if len(gt_Y.shape) == 1:
        n_sample = gt_Y.shape[0]
        n_feature = 1
        for i in range(n_sample):
            for j in range(n_feature):
                output += w[j]*np.abs(pred_Y[i] - gt_Y[i])
    else:
        n_sample = gt_Y.shape[1]
        n_feature = gt_Y.shape[0]
        for i in range(n_sample):
            for j in range(n_feature):
                output += w[j]*np.abs(pred_Y[j][i] - gt_Y[j][i])
    return output/n_sample

def NAE(pred_Y,gt_Y):
    # input : (3 ,num of data)
    output = 0.
    if len(gt_Y.shape) == 1:
        n_sample = gt_Y.shape[0]
        n_feature = 1
        for i in range(n_sample):
            for j in range(n_feature):
                output += np.abs(pred_Y[i] - gt_Y[i])/gt_Y[i]
    else:
        n_sample = gt_Y.shape[1]
        n_feature = gt_Y.shape[0]
        for i in range(n_sample):
            for j in range(n_feature):
                output += np.abs(pred_Y[j][i] - gt_Y[j][i])/gt_Y[j][i]
    return output/n_sample

def main():
    # loading data
    print("<"+"="*50+">")
    start_time = time.time()
    X_train_path = "./data/X_train.npz"
    Y_train_path = "./data/Y_train.npz"
    X_test_path = "./data/X_test.npz"
    print("Loading dataset...")
    X_train_full = load_file(X_train_path)
    Y_train_full = load_file(Y_train_path)
    X_test_full = load_file(X_test_path)
    X_train_full = np.concatenate((X_train_full[:,:MSD_feature_size],X_train_full[:,5000:5000+VAC_feature_size]),axis = 1)
    X_test_full = np.concatenate((X_test_full[:,:MSD_feature_size],X_test_full[:,5000:5000+VAC_feature_size]),axis = 1)

    print("total training data size :",X_train_full.shape)
    print("total training label size :",Y_train_full.shape)
    print("total testing data size :",X_test_full.shape)
    X_train_t ,Y_train_t, X_valid ,Y_valid = dataloader(X_train_full,Y_train_full)

    print("train data size :",X_train_t.shape)
    print("train label size :",Y_train_t.shape)
    print("valid data size :",X_valid.shape)
    print("valid label size :",Y_valid.shape)
    end_time = time.time()
    print("Processing time (sec) : %.4f" %(end_time-start_time))
    print("Loading dataset finished!")

    total_train_pred_Y = []
    total_valid_pred_Y = []
    total_test_pred_Y = []
    for iter in range(T):
        print("==========Iteratation : %d/%d==========" %(iter+1,T))
        # Random sample
        sample_idx = random.sample([i for i in range(train_size)],3000)
        X_train = X_train_t[sample_idx]
        Y_train = Y_train_t[:,sample_idx]

        # fitting model
        print("<"+"="*50+">")
        print("Training RandomForestRegressor model...")
        start_time = time.time()
        reg = []
        reg_pr = RandomForestRegressor(n_estimators=300,max_depth=20, random_state=312)
        reg.append(reg_pr.fit(X_train, Y_train[0]))
        print("reg_pr done!")
        reg_ms = RandomForestRegressor(n_estimators=300,max_depth=20, random_state=312)
        reg.append(reg_ms.fit(X_train, Y_train[1]))
        print("reg_ms done!")
        reg_alp = RandomForestRegressor(n_estimators=300,max_depth=20, random_state=312)
        reg.append(reg_alp.fit(X_train, Y_train[2]))
        print("reg_alp done!")
        end_time = time.time()
        print("Processing time (sec) : %.4f" %(end_time-start_time))



        # predict data
        # TRAIN WMAE/NAE
        print("<"+"="*50+">")
        print("[Ein]")
        train_pred_Y = []
        for i in range(len(reg)):
            print("\rProcessing [%d/%d]"%(i+1,len(reg)),end = "")
            tmp_reg = reg[i]
            train_pred_Y_tmp = tmp_reg.predict(X_train)
            train_pred_Y.append(train_pred_Y_tmp)
        total_train_pred_Y.append(train_pred_Y)
        write_file(train_pred_Y,"./train_pred_Y.csv")
        wmae = WMAE(train_pred_Y,Y_train)
        nae = NAE(train_pred_Y,Y_train)
        print("")
        print("WMAE = ",wmae)
        print("NAE = ",nae)

        # VALID WMAE/NAE
        print("<"+"="*50+">")
        print("[Eout(valid)]")
        valid_pred_Y = []
        for i in range(len(reg)):
            print("\rProcessing [%d/%d]"%(i+1,len(reg)),end = "")
            tmp_reg = reg[i]
            valid_pred_Y_tmp = tmp_reg.predict(X_valid)
            valid_pred_Y.append(valid_pred_Y_tmp)
        total_valid_pred_Y.append(valid_pred_Y)
        write_file(valid_pred_Y,"./valid_pred_Y.csv")
        wmae = WMAE(valid_pred_Y,Y_valid)
        nae = NAE(valid_pred_Y,Y_valid)
        print("")
        print("WMAE = ",wmae)
        print("NAE = ",nae)

        # Testing session
        print("<"+"="*50+">")
        print("[Testing]")
        test_pred_Y = []
        for i in range(len(reg)):
            print("\rProcessing [%d/%d]"%(i+1,len(reg)),end = "")
            tmp_reg = reg[i]
            test_pred_Y_tmp = tmp_reg.predict(X_test_full)
            test_pred_Y.append(test_pred_Y_tmp)
        total_test_pred_Y.append(test_pred_Y)
        write_file(test_pred_Y,output_file_name)
        print("")
        print("Testing file done!")

    print("<"+"="*50+">")
    print("Using average method...")
    avg_train_pred_Y = np.array(total_train_pred_Y).mean(0)
    avg_valid_pred_Y = np.array(total_valid_pred_Y).mean(0)
    avg_test_pred_Y = np.array(total_test_pred_Y).mean(0)
    print("[Training data]")
    write_file(avg_train_pred_Y,"./avg_train_pred_Y.csv")
    wmae = WMAE(avg_train_pred_Y,Y_train)
    nae = NAE(avg_train_pred_Y,Y_train)
    print("WMAE = ",wmae)
    print("NAE = ",nae)
    print("[Validating data]")
    write_file(avg_valid_pred_Y,"./avg_valid_pred_Y.csv")
    wmae = WMAE(avg_valid_pred_Y,Y_valid)
    nae = NAE(avg_valid_pred_Y,Y_valid)
    print("WMAE = ",wmae)
    print("NAE = ",nae)


    print("[Testing data] write!")
    write_file(avg_test_pred_Y,"./RandomForestRegressor_avg.csv")

    print("<"+"="*28+"Done"+"="*28+">")
    return 0

if __name__ == "__main__":
    main()
