
import numpy as np
import libbnp as bnp
from dirHdpGenerative import *


class HDP_var_inc(HDP_base):
  
  def __init__(s,K,T,Nw,omega,alpha,Lambda):
    s.base = Dir()
    s.hdp_var = bnp.HDP_var()
    s.firstEst = True # true until the first estimate has been made
    HDP_base.__init__(s,K,T,Nw,omega,alpha,Lambda)

  def updateEst(x_tr, x_te=None, kappa, S):

    for x_i in x_tr:
      s.hdp_var.addDoc(x_i.astype('uint32'))

    if x_te is not None:
      for x_i in x_te:
        s.hdp_var.addHeldOut(x_i.astype('uint32'))
    
    if s.firstEst:
      print('initial estimate: D={}; K={}; T={}; kappa={}; Nw={}; S={};'.format(len(x_tr),s.K,s.T,kappa,s.Nw,S))
      s.hdp_var.densityEst(s.Nw,kappa,s.K,s.T,S)
    else:
      print('updated estimate: D={}; K={}; T={}; kappa={}; Nw={}; S={};'.format(len(x_tr),s.K,s.T,kappa,s.Nw,S))
      s.hdp_var.updateEst(kappa,S)


