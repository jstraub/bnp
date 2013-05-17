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
import argparse

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



class HDPvar(bnp.HDP_var_Dir):
  # x are data for training; x_ho is held out data
  def initialEstimate(self,x,x_ho,Nw,kappa,K,T,S):
    D = len(x)
    for x_i in x:
      self.addDoc(np.resize(x_i,(1,x_i.size)))
      #self.addDoc(np.vstack(x_i[0:N_d]))
    for x_ho_i in x_ho:
      print("adding held out")
      self.addHeldOut(np.resize(x_ho_i,(1,x_ho_i.size)))
      
      #self.addHeldOut(np.vstack(x_ho_i[0:N_d]))
    return self.densityEst(Nw,kappa,K,T,S)


class HDPgibbs(bnp.HDP_gibbs_Dir):
  # x are data for training; x_ho is held out data
  def initialEstimate(self,x,x_ho,Nw,K0,T0,It):
    D = len(x)
    for x_i in x:
      xx=np.hstack(x_i)
      #print("adding {}; {}".format(xx.shape,xx))
      self.addDoc(x_i.reshape((1,x_i.shape[0])))
      #self.addDoc(np.vstack(x_i[0:N_d]))
    for x_ho_i in x_ho:
      self.addHeldOut(x_i.reshape((1,x_i.shape[0])))
      #self.addHeldOut(np.vstack(x_ho_i[0:N_d]))
    return self.densityEst(Nw,K0,T0,It)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = 'hdp topic modeling of synthetic data')
  parser.add_argument('-T', type=int, default=10, help='document level truncation')
  parser.add_argument('-K', type=int, default=100, help='corpus level truncation')
  parser.add_argument('-S', type=int, default=10, help='mini batch size')
  parser.add_argument('-D', type=int, default=500, help='number of documents to synthesize')
  parser.add_argument('-H', type=int, default=10, help='number of held out documents for perplexity computation')
  parser.add_argument('-N', type=int, default=100, help='number of words per document')
  parser.add_argument('-Nw', type=int, default=40, help='alphabet size (how many different words)')
  parser.add_argument('-a','--alpha', type=float, default=1.0, help='concentration parameter for document level')
  parser.add_argument('-o','--omega', type=float, default=10.0, help='concentration parameter for corpus level')
  parser.add_argument('-k','--kappa', type=float, default=0.9, help='forgetting rate for stochastic updates')
  #parser.add_argument('-s', action='store_false', help='switch to make the program use synthetic data')
  parser.add_argument('-g','--gibbs', action='store_true', help='switch to make the program use gibbs sampling instead of variational')
  args = parser.parse_args()
  print('args: {0}'.format(args))


  D = args.D #number of documents to process
  D_ho = args.H # (ho= held out) number of docs used for testing (perplexity)
  N_d = args.N # max number of words per doc
  Nw = args.Nw # how many different symbols are in the alphabet
  kappa = args.kappa # forgetting rate
  K = args.K # top level truncation
  T = args.T # low level truncation
  S = args.S # mini batch size
  alpha = args.alpha # concentration on G_i
  omega = args.omega # concentration on G_0
  dirAlphas = np.ones(Nw) # alphas for dirichlet base measure

  dirichlet=bnp.Dir(dirAlphas)

  hdp_true = HDP_sample(K,T,Nw,omega,alpha,dirichlet)
  x, gtCorpProp, gtTopic, pi, c = hdp_true.generateDirHDPSample(D+D_ho,N_d)
  x_tr = x[0:D-D_ho]
  x_ho = x[D-D_ho:D]

  hdp_true.save('sample.mat')
  hdp_true.load('sample.mat')

  print("---------------- Starting! use " + str(D) +" docs and " + str(D_ho) + " held out --------------")

  print("Dir created")

  if args.gibbs:
    print('Gibbs');
    It = 100;
    K0 = 30;
    T0 = 10;
    hdp = HDPgibbs(dirichlet,alpha,omega)

    start = time.clock()
    hdp.initialEstimate(x_tr,x_ho,Nw,K0,T0,It)
    print('--- time taken: {}'.format(time.clock() - start))

    perp = np.zeros(D_ho)
    hdp.getPerplexity(perp)
    print('Perplexity of test docs: {}'.format(perp))

#    hdp=bnp.HDP_Dir(dirichlet,alpha,omega)
#    for x_i in x[0:D]:
#      hdp.addDoc(np.vstack(x_i[0:N_d]))
#    result=hdp.densityEst(10,10,10)

  else:
    print('Variational');
    hdp = HDPvar(dirichlet,alpha,omega)

    start = time.clock()
    hdp.initialEstimate(x_tr,x_ho,Nw,kappa,K,T,S)
    print('--- time taken: {}'.format(time.clock() - start))

#    hdp=bnp.HDP_onl(dirichlet,alpha,omega)
#    for x_i in x[0:D]:
#      hdp.addDoc(np.vstack(x_i[0:N_d]))
#    result=hdp.densityEst(Nw,kappa,K,T)

    perp = np.zeros(D-D_ho)
    hdp.getPerplexity(perp)
    print('Perplexity of iterations: {0}'.format(perp))
    
    fig00=plt.figure()
    plt.plot(perp)
    fig00.show()
#    raw_input('Press enter to continue')

#    perp_d=np.zeros(D_ho)
#    for d in range(0,D_ho):
#      print('{0}: {1}'.format(d,x_ho[d]))
#      perp_d[d]=hdp.perplexity(x_ho[d],D-D_ho+1,kappa)
#      print('Perplexity of heldout ({0}):\t{1}'.format(d,perp_d))
#
#    fig01=plt.figure()
#    plt.plot(perp_d)
#    fig01.show()
#    raw_input('Press enter to continue')

    hdp_var = HDP_sample(K,T,Nw,omega,alpha,dirAlphas)
    #hdp_var.loadHDPSample(x,topic,docTopicInd,z,v,sigV,pi,sigPi,omega,alpha,dirAlphas)
    hdp_var.loadHDPSample(x_tr=x_tr,x_te=x_ho,hdp=hdp)
    hdp_var.save('model.mat')
    hdp_var.load('model.mat')

    fig1=plt.figure()
    plt.imshow(hdp_true.docTopicsImg(),interpolation='nearest', cmap = cm.hot)
    fig1.show()
    fig2=plt.figure()
    plt.imshow(hdp_var.docTopicsImg(),interpolation='nearest', cmap = cm.hot)
    fig2.show()
    
    print('{}'.format(hdp_true.state['logP_w']))
    klDvar2true = hdp_var.klD(hdp_var.state['logP_w'],hdp_true.state['logP_w'])
    klDtrue2var = hdp_var.klD(hdp_true.state['logP_w'],hdp_var.state['logP_w'])

    fig3=plt.figure()
    plt.subplot(1,2,0) 
    plt.stem(np.arange(0,D),klDvar2true)
    plt.ylabel('divergence D(var||true)')
    plt.xlabel('documents')
    plt.subplot(1,2,1) 
    plt.stem(np.arange(0,D),klDtrue2var)
    plt.ylabel('divergence D(true||var)')
    plt.xlabel('documents')
    fig3.show()

    print('D(var||true): {}'.format(klDvar2true))
    print('D(true||var): {}'.format(klDtrue2var))
    

    raw_input()


