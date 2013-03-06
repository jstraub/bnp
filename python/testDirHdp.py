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

  D = 1000 #number of documents to process
  N_d = 10 # max number of words per doc
  Nw = 256 # how many different symbols are in the alphabet
  ro = 0.75 # forgetting rate
  K = 40 # top level truncation
  T = 10 # low level truncation
  alpha = 1. # concentration on G_i
  gamma = 10. # concentration on G_0
  dirAlphas = np.ones(Nw)*1.0e-5 # alphas for dirichlet base measure

  pathToData = "../../data/bof/bofs249.txt"
  x=[];
  sceneType=[]
  f = open(pathToData,'r')
  for line in f:
    x_i = np.fromstring(line,dtype=np.uint32,sep='\t')
    # x_i = x_i[1::] # first elem is image indicator
    # TODO sth is wrong with the bofs ... check that!
    x_i = np.delete(x_i,np.nonzero(x_i[1::]>Nw))
    x.append(x_i);
    sceneType.append(x_i[0])

    #print(np.max(x_i))
    #if len(x) == 9 :
    #  print(x_i.T)

  D=min(D,len(x))

  print("---------------- Starting! use " + str(D) +" docs of " + str(len(x)) + "--------------")

  dirichlet=bnp.Dir(dirAlphas)
  print("Dir created")

  variational = False
  if variational:
    hdp=bnp.HDP_onl(dirichlet,alpha,gamma)
    for x_i in x[0:D]:
      hdp.addDoc(np.vstack(x_i[0:N_d]))
    result=hdp.densityEst(Nw,ro,K,T)
  else:
    hdp=bnp.HDP_Dir(dirichlet,alpha,gamma)
    for x_i in x[0:D]:
      hdp.addDoc(np.vstack(x_i[0:N_d]))
    result=hdp.densityEst(10,10,10)


  print("---------------------- DONE -------------------------");

#  #z_di=[]
#  lamb=[]
#  for k in range(0,K):
#    lamb.append(np.zeros(Nw,dtype=np.double))
#    hdp.getLambda(lamb[k],k)
#    print(lamb[k])
#    #z_di.append(np.zeros(len(x[d]),dtype=np.uint32))
#    #hdp.getClassLabels(z_di[d],d)
#    #print(z_di[d])
#  a=np.zeros(K,dtype=np.double)
#  b=np.zeros(K,dtype=np.double)
#  hdp.getA(a)
#  hdp.getB(b)
#
#  plt.figure(1)
#  for k in range(0,40):
#    plt.subplot(7,6,k+1)
#    plt.plot(lamb[k])
#    plt.xlabel('topic '+str(k))
#
#  plt.subplot(7,6,41)
#  plt.plot(a)
#  plt.xlabel('A')
#  plt.subplot(7,6,42)
#  plt.plot(b)
#  plt.xlabel('B')
#  plt.show(1)
#
  
#  plt.figure(1)
#  for d in range(0,J):
#    plt.subplot(1,J,d)
#    for k in range(0,K):
#      plt.plot(x[d][z_di[d]==k,0],x[d][z_di[d]==k,1],'xk',color=cm.spectral(float(k)/(K-1)),hold=True)
#      print(cm.spectral(float(k)/K))
#      plt.xlim([-12,12])
#      plt.ylim([-12,12])
#  plt.show(1)
#
