#!/usr/bin/python

# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or 
# http://www.opensource.org/licenses/mit-license.php 

import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import numpy as np

import time
import cProfile
import argparse

import libbnp as bnp

from dirHdpGenerative import *
from hdpIncremental import *

import fileinput

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = 'hdp topic modeling of synthetic data')
  parser.add_argument('-T', type=int, default=10, help='document level truncation')
  parser.add_argument('-K', type=int, default=100, help='corpus level truncation')
  parser.add_argument('-S', type=int, default=10, help='mini batch size')
  #parser.add_argument('-D', type=int, default=500, help='number of documents to synthesize')
  parser.add_argument('-H', type=int, default=2, help='number of held out documents for perplexity computation')
  parser.add_argument('-N', type=int, default=100, help='number of words per document')
  parser.add_argument('-Nw', type=int, default=10, help='alphabet size (how many different words)')
  parser.add_argument('-a','--alpha', type=float, default=3.0, help='concentration parameter for document level')
  parser.add_argument('-o','--omega', type=float, default=30.0, help='concentration parameter for corpus level')
  parser.add_argument('-k','--kappa', type=float, default=0.9, help='forgetting rate for stochastic updates')
  #parser.add_argument('-s', action='store_false', help='switch to make the program use synthetic data')
  parser.add_argument('-g','--gibbs', action='store_true', help='switch to make the program use gibbs sampling instead of variational')
  args = parser.parse_args()
  print('args: {0}'.format(args))


  #D = args.D #number of documents to process
  D_te = args.H # (ho= held out) number of docs used for testing (perplexity)
  N_d = args.N # max number of words per doc
  Nw = args.Nw # how many different symbols are in the alphabet
  kappa = args.kappa # forgetting rate
  K = args.K # top level truncation
  T = args.T # low level truncation
  S = args.S # mini batch size
  alpha = args.alpha # concentration on G_i
  omega = args.omega # concentration on G_0
  dirAlphas = np.ones(Nw)*1.1 # alphas for dirichlet base measure

  print("---------------- Starting! --------------")
  
  discrete = False
  if discrete:
    dataType='uint32'
    hdp = HDP_var_Dir_inc(K,T,Nw,omega,alpha,dirAlphas)
  else:
    dataType='double'
    hdp = HDP_var_NIW_inc(K,T,Nw,omega,alpha,np.ones((1,1))*(-5),2.1,np.ones((1,1))*8.1*3,2.1)

  x=[]
  x_tr=[]
  x_te=[]
  for line in fileinput.input():
    if len(x_te) < D_te:
      x_te.append(np.fromstring(line, dtype=dataType, sep=" "))
    else:
      x.append(np.fromstring(line, dtype=dataType, sep=" "))
      #print('{}'.format(x[-1]))
    if len(x) >= S:
      print('----------')
  hdp.updateEst(x,kappa,S,x_te)
  x_tr.extend(x)
  x=[]
#    if len(x) >= S+2:
#      print('----------')
#      hdp.updateEst(x[0:-3],kappa,S,x_te=x[-2:-1])
#      x_tr.extend(x[0:-3])
#      x_te.extend(x[-2:-1])
#      x=[]

  hdp.loadHDPSample(x_tr,x_te, hdp.hdp_var)
  #print('{}'.format(hdp.state['logP_w']))
  hdp.save('incTest.mat')

# -- make sure the saved model is saved and loaded correctly
#  hdp2 = HDP_var_Dir_inc(K,T,Nw,omega,alpha,dirAlpha)
  hdp2 = HDP_var_NIW_inc(K,T,Nw,omega,alpha,np.ones((1,1))*5,1,np.ones((1,1)),1)
  hdp2.load('incTest.mat')
  #print('{}'.format(hdp2.state['logP_w']))
  hdp.stateEquals(hdp2)

# -- plot some things
  fig0=plt.figure(0)
  imgplot=plt.imshow(hdp.docTopicsImg(),interpolation='nearest',cmap=cm.hot);
  fig0.show()

  if not discrete:

    X=np.linspace(-20,20,100)
    sig=np.sqrt(hdp.state['beta'][:,0])
    mu=hdp.state['beta'][:,1]

    fig1=plt.figure()
    prop = hdp.state['sigV']
    for i in range(0,mu.shape[0]):
      plt.plot(X,prop[i]*mlab.normpdf(X,mu[i],sig[i]))
    fig1.show()

    toPlot = [0,1,2,3,4,100,101,102,103,104]
    fig2=plt.figure()
    for d in toPlot:
      sigPi=hdp.state['sigPi'][d,:]
      c = hdp.state['c'][d,:]
      
      Y=np.zeros(X.size)
      c_u = np.unique(c)
      sigPi_u = np.zeros(c_u.size)
      for i in range(0,c_u.size):
        sigPi_u[i] = np.sum(sigPi[c_u[i] == c])
        Y += sigPi_u[i]*mlab.normpdf(X,mu[c_u[i]],sig[c_u[i]])
      print('{}: c_u={}; sigPi_u={}'.format(d,c_u,sigPi_u))
      
      plt.subplot(2,1,d/100)
      plt.plot(X,Y)
      plt.title(str(d))
    
      plt.plot(X,mlab.normpdf(X,np.mean(x_tr[d]), np.std(x_tr[d]) ),'r-')

    fig2.show()
      
    


#  symKLimg = hdp.symKLImg();
#  print('logP_w:\n{}'.format(hdp.state['logP_w']))
#  print('symKLimg:\n{}'.format(symKLimg))
#  fig1=plt.figure(1)
#  imgplot=plt.imshow(symKLimg,interpolation='nearest',cmap=cm.hot);
#  plt.colorbar()
#  fig1.show()
#
#  jsDimg = hdp.jsDImg();
#  print('jsDimg:\n{}'.format(jsDimg))
#  fig2=plt.figure(2)
#  imgplot=plt.imshow(jsDimg,interpolation='nearest',cmap=cm.hot);
#  plt.colorbar()
#  plt.show()

  time.sleep(10000)

