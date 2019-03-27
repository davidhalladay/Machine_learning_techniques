###########################################
# FileName     [ problem_02.py ]
# Synopsis     [ hard-margin support vector machine algorithm ]
# Author       [ Wan-Cyuan Fan ]
###########################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


x = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
y = np.array([-1,-1,-1,1,1,1,1])

clf = SVC(kernel = 'poly' , degree = 2, coef0 = 1, gamma = 1)
clf.fit(x, y)

print(clf.support_)
print("support vectors : ",clf.support_vectors_)
print("Alpha * y: ",clf.dual_coef_)

#find b
b = []
for i in clf.support_:
    tmp = 0.
    for j , k in zip(clf.support_,range(0,5)):
        tmp += clf.dual_coef_[0][k] * (1+np.inner(x[j],x[i]))**2.
    b.append(y[i] - tmp)
b = np.array(b)
print("b : ",b)

#find w
w = []
for j , k in zip(clf.support_,range(0,5)):
    ay = clf.dual_coef_[0][k]
    tmp = [ay*x[j][0]**2.,ay*x[j][1]**2.,ay*2*x[j][0],ay*2*x[j][1],ay*1]
    tmp = np.array(tmp)
    w.append(tmp)
w = np.array(w)
w = np.sum(w,axis = 0)
print("w : ",w)

# plot the image
plt.figure(figsize = (10,8))

for i in range(3):
    plt.scatter(x[i][0], x[i][1], marker='x',
            color='blue', alpha=0.7, label='x')
for i in range(3,7):
    plt.scatter(x[i][0], x[i][1], marker='o',
            color='blue', alpha=0.7, label='x')
def f(x1,x2):
    return 0.8887*x1**2 + 0.6665544*x2**2 - 1.7774*x1 +2.22044605e-16*x2 -1.66655442

xx = np.linspace(-2, 2, 1000)
yy = np.linspace(-2, 2, 1000)
YY, XX = np.meshgrid(yy, xx)
plt.contour(XX, YY, f(XX,YY), colors='k',levels=[0], alpha=0.5,
           linestyles='-')
plt.savefig("./problem_contour.png")
