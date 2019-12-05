import sys
sys.path.append("../")

from kernel import InverseMultiquadricKernelPytouch
from mtgp import MultiTaskGaussianProcess

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.style.use("ggplot")


N1 = 10
N2 = 50
X1 = np.linspace(0, np.pi*2, N1)[:,None]
X2 = np.linspace(0, np.pi*4, N2)[:,None]
Y1 = 2*np.sin(X1) + np.random.randn(N1)[:,None] * 0.2
Y2 = np.sin(X2) + np.random.randn(N2)[:,None] * 0.2

X1 = torch.from_numpy(X1).float()
X2 = torch.from_numpy(X2).float()
Y1 = torch.from_numpy(Y1).float()
Y2 = torch.from_numpy(Y2).float()

kern = InverseMultiquadricKernelPytouch([0.4])
model = MultiTaskGaussianProcess([X1,X2], [Y1,Y2], [0,1], kern)


xx = np.linspace(0, np.pi*4, 200)[:,None]
xx = torch.from_numpy(xx).float()
mm1, ss1 = model.predict(xx, 0)
mm2, ss2 = model.predict(xx, 1)

mm1 = mm1.numpy().ravel()
ss1= np.sqrt(ss1.numpy().ravel())
mm2 = mm2.numpy().ravel()
ss2= np.sqrt(ss2.numpy().ravel())
xx = xx.numpy().ravel()
X1 = X1.numpy().ravel()
X2 = X2.numpy().ravel()
Y1 = Y1.numpy().ravel()
Y2 = Y2.numpy().ravel()

line1 = plt.plot(X1, Y1, "*")
line2 = plt.plot(X2, Y2, "*")
plt.plot(xx, mm1, color=line1[0].get_color())
plt.fill_between(xx, mm1+ss1, mm1-ss1, color=line1[0].get_color(), alpha=0.3)
# plt.plot(xx, mm1+ss1, "--", color=line1[0].get_color())
# plt.plot(xx, mm1-ss1, "--", color=line1[0].get_color())
plt.plot(xx, mm2, color=line2[0].get_color())
plt.fill_between(xx, mm2+ss2, mm2-ss2, color=line2[0].get_color(), alpha=0.3)
# plt.plot(xx, mm2+ss2, "--", color=line2[0].get_color())
# plt.plot(xx, mm2-ss2, "--", color=line2[0].get_color())
plt.show()

######################
print(model.task_params_to_psd())
model.learning()
print(model.task_params_to_psd())

xx = np.linspace(0, np.pi*4, 100)[:,None]
xx = torch.from_numpy(xx).float()
mm1, ss1 = model.predict(xx, 0)
mm2, ss2 = model.predict(xx, 1)

mm1 = mm1.numpy().ravel()
ss1= np.sqrt(ss1.numpy().ravel())
mm2 = mm2.numpy().ravel()
ss2= np.sqrt(ss2.numpy().ravel())
xx = xx.numpy().ravel()

line1 = plt.plot(X1, Y1, "*")
line2 = plt.plot(X2, Y2, "*")
plt.plot(xx, mm1, color=line1[0].get_color())
plt.fill_between(xx, mm1+ss1, mm1-ss1, color=line1[0].get_color(), alpha=0.3)
# plt.plot(xx, mm1+ss1, "--", color=line1[0].get_color())
# plt.plot(xx, mm1-ss1, "--", color=line1[0].get_color())
plt.plot(xx, mm2, color=line2[0].get_color())
plt.fill_between(xx, mm2+ss2, mm2-ss2, color=line2[0].get_color(), alpha=0.3)
# plt.plot(xx, mm2+ss2, "--", color=line2[0].get_color())
# plt.plot(xx, mm2-ss2, "--", color=line2[0].get_color())
plt.show()

            # try:
    #     invK = torch.inverse(K)
    # except:
    #     print 'hogehoge'
    #     X = self.X
    #     sigma = torch.exp(self.sigma)
    #     import ipdb; ipdb.set_trace()
    # invL = torch.solve(torch.eye(L.size()[0]), L)[0]
    # invK = torch.mm(invL.T, invL)