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
    D = 100 #number of documents to process
    N_d = 100 # max number of words per doc
    Nw = 4 # how many different symbols are in the alphabet
    ro = 0.9 # forgetting rate
    K = 10 # top level truncation
    T = 3 # low level truncation
    alpha = 1. # concentration on G_i
    omega = 10. # concentration on G_0
    dirAlphas = np.ones(Nw) # alphas for dirichlet base measure

    hdp_sample = HDP_sample(omega,alpha,dirAlphas)

    x, gtCorpProp, gtTopic, pi, c = hdp_sample.generateDirHDPSample(D,N_d,K,T,Nw)

    logP_gt = hdp_sample.logP_self()
    print('logP of groundtruth = {}'.format(logP_gt))

  else:
    D = 1000 #number of documents to process
    N_d = 10 # max number of words per doc
    Nw = 256 # how many different symbols are in the alphabet
    ro = 0.75 # forgetting rate
    K = 40 # top level truncation
    T = 10 # low level truncation
    alpha = 1. # concentration on G_i
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
  else:
    hdp=bnp.HDP_Dir(dirichlet,alpha,omega)
    for x_i in x[0:D]:
      hdp.addDoc(np.vstack(x_i[0:N_d]))
    result=hdp.densityEst(10,10,10)

  print("---------------------- Corpus Topic Proportions -------------------------");

  corpProp = np.zeros(K,dtype=np.double)
  hdp.getCorpTopicProportions(corpProp)
  print('topic proportions: \t{}\t{}'.format(corpProp,np.sum(corpProp)))
  print('GT topic proportions: \t{}\t{}'.format(gtCorpProp,np.sum(gtCorpProp)))

  print("---------------------- Corpus Topics -------------------------");
  topic=[]
  for k in range(0,K):
    topic.append(np.zeros(Nw,dtype=np.double))
    hdp.getCorpTopic(topic[k],k)
    print('topic_{}=\t{}\t{}'.format(k,topic[k]))
    print('gtTopic_{}=\t{}\t{}'.format(k,gtTopic[k,:]))
#  a=np.zeros(K,dtype=np.double)
#  b=np.zeros(K,dtype=np.double)
#  hdp.getA(a)
#  hdp.getB(b)
#  print('alpha_1={}'.format(a))
#  print('alpha_2={}'.format(b))

  docProp=[]
  docTopicInd=[]
  for d in range(0,D):
    docProp.append(np.zeros(T,dtype=np.double))
    docTopicInd.append(np.zeros(T,dtype=np.uint32))
    hdp.getDocTopics(docProp[d],docTopicInd[d],d)
    print('docTopicProp({}): {}'.format(d,docProp[d]))
    print('docTopicInd({}): {}'.format(d,docTopicInd[d]))

  vT = np.zeros((K,D))
  for d in range(0,D):
    for t in range(0,T):
      k=docTopicInd[d][t]
      vT[k,d]=docProp[d][t]





