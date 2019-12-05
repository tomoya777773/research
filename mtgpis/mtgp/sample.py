import numpy as np
import torch

class MTGPRegression:
  def __init__(self, X, Y, T, kernel):
    self.X = torch.cat(X)
    self.Y = torch.cat(Y)
    self.T = torch.cat([torch.ones(X[i].shape[0],1, dtype=torch.long)*T[i] for i in range(len(X))])
    self.kern = kernel
    self.sigma = torch.tensor([np.log(0.3)])
    
    self.N_task = self.T.unique().shape[0]
    self.task_kern_param = torch.rand(self.N_task, self.N_task)



  def task_K(self):
    u = torch.triu(torch.exp(self.task_kern_param))
    return u.t().mm(u)


  def predict(self, x, t):
    # Kx = self.kern.K(x)
    Kx  = ( torch.ones(len(x)) * (1.0 / float(self.kern.params[0])) )[:, None]
    KXx = self.kern(self.X, x)
    KX = self.kern(self.X, self.X)

    t = torch.ones(x.shape[0], 1, dtype=torch.long) * t

    task_kern = self.task_K()
    Kt = task_kern[t, t]
    KTt = task_kern[self.T, t.t()]
    KT = task_kern[self.T, self.T.t()]
    
    sig = torch.exp(self.sigma)

    mean = (KTt.t()*KXx.t()).mm(torch.solve(self.Y, KT*KX+torch.eye(KX.shape[0])*sig)[0])
    sigma = torch.diag(Kt*Kx - (KTt.t()*KXx.t()).mm(torch.solve(KTt*KXx, KT*KX+torch.eye(KX.shape[0])*sig)[0])).reshape(x.shape[0], -1)

    return mean, sigma
    
  
  def compute_grad(self, flag):
    self.sigma.requires_grad = flag
    self.task_kern_param.requires_grad = flag
    self.kern.params.requires_grad = flag


  def negative_log_likelihood(self):
    task_kern = self.task_K()
    # KT = [[task_kern[int(x)][int(y)] for x in self.T.flatten()] for y in self.T.flatten()]
    KT = task_kern[self.T, self.T.t()]
    K = KT*self.kern(self.X, self.X) + torch.eye(self.X.shape[0])*torch.exp(self.sigma)

        
    invKY = torch.solve(self.Y, K)[0]
    # logdet = torch.cholesky(K, upper=False).diag().log().sum()
    logdet = torch.logdet(K)

    return (logdet + self.Y.t().mm(invKY))
    

  def learning(self):
    max_iter = 100

    self.compute_grad(True)
    param = [self.kern.params] + [self.sigma] + [self.task_kern_param]
    optimizer = torch.optim.Adam(param, lr=0.01)
    optimizer = torch.optim.LBFGS(param, history_size=20, max_iter=20)

    for i in range(max_iter):
      optimizer.zero_grad()
      f = self.negative_log_likelihood() 
      f.backward()
      print(f.item())
      print("-------")
      def closure():
        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
          f = self.negative_log_likelihood() 
          f.backward()
        return f
      optimizer.step(closure)
      # optimizer.step()
    self.compute_grad(False)
    

if __name__=="__main__":
  import sys
  from kernel import InverseMultiquadricKernelPytouch
  import matplotlib.pyplot as plt
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
  model = MTGPRegression([X1,X2], [Y1,Y2], [0,1], kern)


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
  print(model.task_K())
  model.learning()
  print(model.task_K())

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