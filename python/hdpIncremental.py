
import numpy as np
import libbnp as bnp
from dirHdpGenerative import *


class HDP_var_inc(HDP_base):
  
  def __init__(s,K,T,Nw,omega,alpha,Lambda):
    s.base = bnp.Dir(Lambda)
    s.hdp_var = bnp.HDP_var(s.base,alpha,omega)
    s.firstEst = True # true until the first estimate has been made
    HDP_base.__init__(s,K,T,Nw,omega,alpha,Lambda)

  def updateEst(s,x_tr, kappa, S, x_te=None):

    for x_i in x_tr:
      s.hdp_var.addDoc(np.resize(x_i,(1,x_i.size)))

    if x_te is not None:
      for x_i in x_te:
        s.hdp_var.addHeldOut(np.resize(x_i,(1,x_i.size)))
    
    if s.firstEst:
      print('initial estimate: D={}; K={}; T={}; kappa={}; Nw={}; S={};'.format(len(x_tr),s.state['K'],s.state['T'],kappa,s.state['Nw'],S))
      s.hdp_var.densityEst(s.state['Nw'],kappa,s.state['K'],s.state['T'],S)
    else:
      print('updated estimate: D={}; K={}; T={}; kappa={}; Nw={}; S={};'.format(len(x_tr),s.state['K'],s.state['T'],kappa,s.state['Nw'],S))
      s.hdp_var.updateEst(kappa,S)


