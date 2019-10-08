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
        for i in range(0,num):
            left_Y = np.array(Y_sort[:i])
            right_Y = np.array(Y_sort[i:])
            #print("left_Y",left_Y.shape)
            #print("right_Y",right_Y.shape)
            err = left_Y.shape[0] * Gini_index(left_Y) + right_Y.shape[0] * Gini_index(right_Y)
            if err < min_err:
                min_err = err
                best_d = d
                best_theta = theta[i]
                best_index = i

    left_X , right_X = test_split(best_d,best_theta,X,Y)
    left_Y , right_Y = test_split(best_d,best_theta,X,Y)
    left_X = left_X.reshape(-1,1)
    right_X = right_X.reshape(-1,1)
    return (left_X, left_Y), (right_X, right_Y), best_d, best_theta

def Gini_index(label):
    #print("label ",label)
    N = label.shape[0]

    T, F = sum(label == 1), sum(label == -1)
    #print(T,F)
    if N == 0 or T == 0 or F == 0:
        return 0.
    else:
        return 1. - ((T / N)**2 + (F / N)**2)

def predict(node, row):
    if node.left == None and node.right == None:
        return node.sign
    else:
        dim, theta = node.dim, node.theta
        if row[dim] < theta:
            return predict(node.left, row)
        else:
            return predict(node.right, row)
def build_Tree(X, Y, depth):

    # termination
    if X.shape[0] == 0: return None

    # when we touch the leaf full growth
    if Gini_index(Y) == 0:
        node = Node(-1, -1)
        node.sign = np.sign(Y[0])
        return node
    # still rec.
    else:
        (left_x, left_y), (right_x, right_y), dim, theta = get_split(X, Y)
        node = Node(dim, theta)
        node.left = build_Tree(left_x, left_y, depth+1)
        node.right = build_Tree(right_x, right_y, depth+1)
        return node
