from gpis import GaussianProcessImplicitSurface
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 6, 0.1)
y = np.sin(x)
position_data = np.zeros((len(x), 2))
for i in range(len(x)):
    position_data[i][0] = x[i]
    position_data[i][1] = y[i]

# position_data = np.append(position_data, [[1, 1.5]], axis=0)
# print position_data

label_data = np.zeros(len(x)).reshape(-1, 1)
# label_data = np.append(label_data, [])
# print label_data

# current_position = np.array([1,1])

# gpis = GaussianProcessImplicitSurface(position_data, label_data)
# n, d = gpis.direction_func(current_position)
# print n,d





"""search"""
n = 30

x_ = np.linspace(1, 5, n)
y_ = np.sin(x_)

true_x, true_y, false_x, false_y, la_list = [],[],[],[],[]
count, true_count, false_count = 0,0,0
X = position_data
Y = label_data

for i in range(n):
    x_sample = np.linspace(x_[i] - 1, x_[i] + 1, n)
    y_sample = np.linspace(y_[i] - 1, y_[i] + 1, n)

    # print(X[np.where((X[:, 0] > x_[i]-0.5) & (X[:, 0] < x_[i]+0.5))])
    # print(Y[np.where((X[:, 0] > x_[i]-0.2) & (X[:, 0] < x_[i]+0.2))])

    X_ = X[np.where((X[:, 0] > x_[i]-1) & (X[:, 0] < x_[i]+1))]
    Y_ = Y[np.where((X[:, 0] > x_[i]-1) & (X[:, 0] < x_[i]+1))]

    fp = GaussianProcessImplicitSurface(X_, Y_)
    la = fp.decision_func([x_sample, y_sample], x_[i])

    la_list.append(la)

    if (la > 0 and y_[i] < 0) or (la <= 0 and y_[i] > 0):
        true_x.append(x_[i])
        true_y.append(y_[i])
        true_count+=1
    else:
        false_x.append(x_[i])
        false_y.append(y_[i])
        false_count+=1
    count += 1
    print("count:", count)

# print(la_list)
print("true count:", true_count)
print("false count:", false_count)
print('true accurency:', true_count/n)
print('false accurency:', false_count/n)


plt.scatter(true_x, true_y, marker="x", color='r', alpha=0.8, label = "correct",s=70)
plt.scatter(false_x, false_y, marker="v", color='b', alpha=0.8, label = "incorrect",s=70)
plt.scatter(x, y, marker='o', color='black',label="sample",s=30)

plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='upper left',fontsize=16)

# plt.savefig('d2_point20.png')
plt.show()
