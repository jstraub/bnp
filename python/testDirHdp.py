#!/usr/bin/python

# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or 
# http://www.opensource.org/licenses/mit-license.php 

import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import time
import cProfile

import libbnp as bnp

from dirHdpGenerative import *

def dataFromBOFs(pathToData):
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
  return x


if __name__ == '__main__':

  useSynthetic = True
  variational = True

  if useSynthetic:
    D = 3 #number of documents to process
    N_d = 100 # max number of words per doc
    Nw = 40 # how many different symbols are in the alphabet
    ro = 0.9 # forgetting rate
    K = 30 # top level truncation
    T = 10 # low level truncation
    alpha = 1. # concentration on G_i
    omega = 10. # concentration on G_0
    dirAlphas = np.ones(Nw) # alphas for dirichlet base measure

    hdp_sample = HDP_sample(K,T,Nw,omega,alpha,dirAlphas)
    x, gtCorpProp, gtTopic, pi, c = hdp_sample.generateDirHDPSample(D,N_d)

  else:
    D = 1000 #number of documents to process
    N_d = 10 # max number of words per doc
    Nw = 256 # how many different symbols are in the alphabet
    ro = 0.75 # forgetting rate
    K = 40 # top level truncation
    T = 10 # low level truncation
    alpha = 1.1 # concentration on G_i
    omega = 10. # concentration on G_0
    dirAlphas = np.ones(Nw)*1.0e-5 # alphas for dirichlet base measure
 
    pathToData = "../../data/bof/bofs249.txt"
    x = dataFromBOFs(pathToData)

  D=min(D,len(x))

  print("---------------- Starting! use " + str(D) +" docs of " + str(len(x)) + "--------------")

  dirichlet=bnp.Dir(dirAlphas)
  print("Dir created")

  if variational:
    hdp=bnp.HDP_onl(dirichlet,alpha,omega)
    for x_i in x[0:D]:
      hdp.addDoc(np.vstack(x_i[0:N_d]))
    result=hdp.densityEst(Nw,ro,K,T)


    hdp_var = HDP_sample(K,T,Nw,omega,alpha,dirAlphas)
    #hdp_var.loadHDPSample(x,topic,docTopicInd,z,v,sigV,pi,sigPi,omega,alpha,dirAlphas)
    hdp_var.loadHDPSample(x=x,hdp=hdp)

    logP_gt = hdp_sample.logP_fullJoint()
    logP_var = hdp_var.logP_fullJoint()
    kl_pq = hdp_sample.KLdivergence(hdp_var)
    kl_qp = hdp_var.KLdivergence(hdp_sample)

    hdp_sample.checkSticks()
    hdp_var.checkSticks()

    print('\n-----------------------------\n')
    print('logP of full joint of groundtruth = {}'.format(logP_gt))
    print('logP of full joint of variational = {}'.format(logP_var))
    print('KL(p||q) = {}'.format(kl_pq))
    print('KL(q||p) = {}'.format(kl_qp))

    fig1=plt.figure()
    plt.imshow(hdp_sample.docTopicsImg(),interpolation='nearest', cmap = cm.hot)
    fig1.show()
    fig2=plt.figure()
    plt.imshow(hdp_var.docTopicsImg(),interpolation='nearest', cmap = cm.hot)
    fig2.show()
    raw_input()

  else:
    hdp=bnp.HDP_Dir(dirichlet,alpha,omega)
    for x_i in x[0:D]:
      hdp.addDoc(np.vstack(x_i[0:N_d]))
    result=hdp.densityEst(10,10,10)



