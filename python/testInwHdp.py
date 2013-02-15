#!/usr/bin/python

# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or 
# http://www.opensource.org/licenses/mit-license.php 

import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import cProfile

import libbnp as bnp

if __name__ == '__main__':
  N=30
  x=[];
  x.append(np.zeros((N*3,2)))
  x[0][0:N,:]=np.random.randn(N,2)+8
  x[0][N:2*N,:]=np.random.randn(N,2)-8
  x[0][2*N::,:]=np.random.randn(N,2)
  x.append(np.zeros((N*3,2)))
  x[1][0:N,:]=np.random.randn(N,2)+8
  x[1][N:2*N,:]=np.random.randn(N,2)
  x[1][2*N::,:]=np.random.randn(N,2)
  x.append(np.zeros((N*3,2)))
  x[2][0:N,:]=np.random.randn(N,2)-8
  x[2][N:2*N,:]=np.random.randn(N,2)
  x[2][2*N::,:]=np.random.randn(N,2)
  J=len(x)

  #mat=sio.loadmat('../../workspace_matlab/dirichlet/testDataEasy.mat')
  #x=mat['x']#.transpose();

  vtheta = np.array([[0.],[0.]])
  kappa = 1. #
  Delta = np.array([[2.,0.],[0.,2.]])
  nu = np.size(x,0)+1.1 #
  alpha = 0.001 # concentration on G_i
  gamma = 100000. # concentration on G_0

  inw=bnp.INW(vtheta,kappa,Delta,nu)
  print("INW created")
  hdp=bnp.HDP_INW(inw,alpha,gamma)
  for x_i in x:
    hdp.addDoc(x_i)
  result=hdp.densityEst(10,10,50)
  z_ji=[]
  K=0
  for j in range(0,J):
    z_ji.append(np.zeros(len(x[j]),dtype=np.uint32))
    hdp.getClassLabels(z_ji[j],j)
    print(z_ji[j])
    K_i=np.max(z_ji[j])
    K=np.max([K_i, K])
  print(K)

  plt.figure(1)
  for j in range(0,J):
    plt.subplot(1,J,j)
    for k in range(0,K):
      plt.plot(x[j][z_ji[j]==k,0],x[j][z_ji[j]==k,1],'xk',color=cm.spectral(float(k)/(K-1)),hold=True)
      print(cm.spectral(float(k)/K))
      plt.xlim([-12,12])
      plt.ylim([-12,12])
  plt.show(1)



