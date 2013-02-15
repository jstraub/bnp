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

  D = 100 #number of documents to process
  N_d = 1000 # max number of words per doc
  Nw = 256 # how many different symbols are in the alphabet
  ro = 0.75 # forgetting rate
  K = 100 # top level truncation
  T = 10 # low level truncation
  alpha = 1. # concentration on G_i
  gamma = 10. # concentration on G_0
  dirAlphas = np.ones(Nw) # alphas for dirichlet base measure

  pathToData = "../../data/bof/bofs249.txt"
  x=[];
  
  f = open(pathToData,'r')
  for line in f:
    x_i = np.fromstring(line,dtype=np.uint32,sep='\n')
    # x_i = x_i[1::] # first elem is image indicator
    # TODO sth is wrong with the bofs ... check that!
    x_i = np.delete(x_i,np.nonzero(x_i[1::]>Nw))

    x.append(x_i);

    #print(np.max(x_i))
    #if len(x) == 9 :
    #  print(x_i.T)

  dirichlet=bnp.Dir(dirAlphas)
  print("Dir created")
  hdp=bnp.HDP_onl(dirichlet,alpha,gamma)
  for x_i in x[0:D]:
    hdp.addDoc(np.vstack(x_i[0:N_d]))
  result=hdp.densityEst(Nw,ro,K,T)


  z_di=[]
  for d in range(0,D):
    z_di.append(np.zeros(len(x[d]),dtype=np.uint32))
    hdp.getClassLabels(z_di[d],d)
    print(z_di[d])

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
