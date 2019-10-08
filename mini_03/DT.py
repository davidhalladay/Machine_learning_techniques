import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random

# reference : https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

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

class Node():
    def __init__(self, dim, theta):
        self.left = None
        self.right = None
        self.dim = dim
        self.theta = theta
        self.sign = 0

def Gini_index(label):
    #print("label ",label)
    N = label.shape[0]
    #print(label)
    T, F = sum(label == 1), sum(label == -1)
    #print(T,F)
    if N == 0 or T == 0 or F == 0:
        return 0.
    else:
        return 1. - ((T / N)**2 + (F / N)**2)

# Split a dataset based on an attribute and an attribute value
def test_split(index,theta, X,Y):
    left_mask = X[:, index] < theta
    right_mask = X[:, index] >= theta
    return X[left_mask], Y[left_mask] , X[right_mask], Y[right_mask]

def get_split(X , Y):
    (num, D) = X.shape
    min_err = np.inf
    best_d = 0. ; best_theta = 0. ; best_index = 0.
    for d in range(D):
        index  = np.argsort(X[:, d])
        X_sort = X[index][:, d]
        Y_sort = Y[index]

        tmp_theta = np.hstack((-np.inf,X_sort))
        tmp_theta = np.delete(tmp_theta,-1)
        theta = ( tmp_theta + X_sort ) / 2.
        theta = np.hstack((theta,np.inf))
        #print(theta)
        for i in range(1,num):
            left_Y = np.array(Y_sort[:i])
            right_Y = np.array(Y_sort[i:])
            # print("left_Y",left_Y.shape)
            # print("right_Y",right_Y.shape)
            err = left_Y.shape[0] * Gini_index(left_Y) + right_Y.shape[0] * Gini_index(right_Y)
            if err < min_err:
                min_err = err
                best_d = d
                best_theta = theta[i]
                best_index = i

    left_X,left_Y,right_X,right_Y = test_split(best_d,best_theta,X,Y)
    return (left_X, left_Y), (right_X, right_Y), best_d, best_theta


# start to create the tree
def build_Tree(X, Y, depth , max_depth):

    # termination
    if X.shape[0] == 0: return None
    if depth > max_depth:
        return None
    # when we touch the leaf full growth
    if Gini_index(Y) == 0:
        node = Node(-1, -1)
        node.sign = np.sign(Y[0])
        return node
    # still rec.
    else:
        (left_x, left_y), (right_x, right_y), dim, theta = get_split(X, Y)

        node = Node(dim, theta)
        node.left = build_Tree(left_x, left_y, depth + 1, max_depth)
        node.right = build_Tree(right_x, right_y, depth + 1, max_depth)
        return node

# Print a decision tree
def print_tree(node, depth = 0):
    if node == None: return 0
    if node.left == None and node.right == None:
        print('    ' * depth + 'leaf: %d' % node.sign)
        return 0

    # first depth
    if node.left != None:
        print('l %s[X%d < %.3f]' % ((depth * ' ', (node.dim ), node.theta)))
        print_tree(node.left, depth + 1)
    if node.right != None:
        print('r %s[X%d < %.3f]' % ((depth * ' ', (node.dim ), node.theta)))
        print_tree(node.right, depth + 1)

    print('%s[%s]' % ((depth * ' ', "x")))

def predict(node, row):
    if node.left == None and node.right == None:
        return node.sign
    else:
        dim, theta = node.dim, node.theta
        if row[dim] < theta:
            return predict(node.left, row)
        else:
            return predict(node.right, row)

def main():
    print("Problem 11:")
    filename = ["./data/hw3_train.dat","./data/hw3_test.dat"]
    train_data ,train_label = load_file(filename[0])
    test_data ,test_label = load_file(filename[1])
    print('train_data shape:', train_data.shape)
    print('test_data shape:', test_data.shape)

    # setting decision tree
    root = build_Tree(train_data, train_label, 0 , np.inf)
    print_tree(root, 0)

    print("\nProblem 12:")
    train_Num = train_data.shape[0]
    test_Num = test_data.shape[0]
    train_pred, test_pred = [], []
    for x in train_data:
        train_pred.append( predict(root, x) )
    for x in test_data:
        test_pred.append( predict(root, x) )
    train_pred = np.array(train_pred)
    test_pred = np.array(test_pred)

    Ein = np.sum( train_pred != train_label ) / train_Num
    Eout = np.sum( test_pred != test_label ) / test_Num
    print('Ein: %f, Eout: %f' % (Ein, Eout))

    print("\nProblem 13:")
    # total depth = 5
    Ein_t = []
    Eout_t = []
    for max_depth in range(5,0,-1):
        root = build_Tree(train_data, train_label, 0 , max_depth)
        train_Num = train_data.shape[0]
        test_Num = test_data.shape[0]
        train_pred, test_pred = [], []
        for x in train_data:
            train_pred.append( predict(root, x) )
        for x in test_data:
            test_pred.append( predict(root, x) )
        train_pred = np.array(train_pred)
        test_pred = np.array(test_pred)

        Ein = np.sum( train_pred != train_label ) / train_Num
        Eout = np.sum( test_pred != test_label ) / test_Num
        Ein_t.append(Ein)
        Eout_t.append(Eout)
        print('Ein: %f, Eout: %f' % (Ein, Eout))
    h = [i for i in range(5,0,-1)]
    plt.figure(figsize = (8,6))
    plt.ylabel('Ein(blue)/Eout(red)')
    plt.xlabel('h')
    plt.plot(h,Ein_t,'-o')
    plt.plot(h,Eout_t,'r-o')
    plt.savefig('E_h.png')

    print("\nProblem 14:")
    iteration = 30000
    train_data ,train_label = load_file(filename[0])
    test_data ,test_label = load_file(filename[1])
    Ein_gt = []
    gt = []
    numofdata = train_data.shape[0]*0.8
    train_pred_gt = []
    test_pred_gt = []
    for T in range(iteration):
        if T%100 == 0:
            print("\r%d/%d" %(T,iteration) , end="")
        idx = np.random.choice(80, size=100)
        rand_train_data ,rand_train_label = train_data[idx] ,train_label[idx]
        root = build_Tree(rand_train_data, rand_train_label, 0 , np.inf)
        # predict
        train_Num = train_data.shape[0]
        test_Num = test_data.shape[0]
        train_pred, test_pred = [], []
        for x in train_data:
            train_pred.append( predict(root, x) )
        for x in test_data:
            test_pred.append( predict(root, x) )
        train_pred = np.array(train_pred)
        Ein = np.sum( train_pred != train_label ) / train_Num
        Ein_gt.append(Ein)
        gt.append(root)
        train_pred_gt.append(train_pred)
        test_pred_gt.append(test_pred)
        if T%500 == 0:
            plt.figure()
            plt.hist(Ein_gt)
            plt.xlabel('t')
            plt.ylabel('Ein_t')
            plt.title('t vs. Ein_t')
            plt.savefig('Ein_hist_30000.png')

    print("finish training")
    plt.figure()
    plt.hist(Ein_gt)
    plt.xlabel('t')
    plt.ylabel('Ein_t')
    plt.title('t vs. Ein_t')
    plt.savefig('Ein_hist_30000.png')

    print("\nProblem 15:")
    acc_g = []
    tot_Ein_Gt = []
    for g in train_pred_gt:
        acc_g.append(g)
        Gt_pred = np.array(acc_g).sum(0)
        Ein_Gt = np.sum( np.sign(Gt_pred) != train_label ) / train_Num
        tot_Ein_Gt.append(Ein_Gt)
    t = [i for i in range(0,iteration)]
    plt.figure()
    plt.plot(t,tot_Ein_Gt)
    plt.xlabel('t')
    plt.ylabel('Ein_Gt')
    plt.title('t vs. Ein_Gt')
    plt.savefig('Ein_Gt_30000.png')

    print("\nProblem 16:")
    acc_g = []
    tot_Eout_Gt = []
    for g in test_pred_gt:
        acc_g.append(g)
        Gt_pred = np.array(acc_g).sum(0)
        Eout_Gt = np.sum( np.sign(Gt_pred) != test_label ) / test_Num
        tot_Eout_Gt.append(Eout_Gt)
    t = [i for i in range(0,iteration)]
    plt.figure()
    plt.plot(t,tot_Eout_Gt)
    plt.xlabel('t')
    plt.ylabel('Eout_Gt')
    plt.title('t vs. Eout_Gt')
    plt.savefig('Eout_Gt_30000.png')

if __name__ == "__main__":
    main()
