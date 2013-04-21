
import numpy as np
import libbnp as bnp
import scipy.special as scisp
import scipy.io as sio

import matplotlib.pyplot as plt

def stickBreaking(v):
  N = v.size
  prop = np.zeros(N+1)
  for i in range(0,N):
    if i == N-1:
      prop[i] = 1.0
    else:
      prop[i]=v[i]
    for j in range(0,i):
      prop[i] *= (1.0-v[j])
  return prop

def logCat(x, pi):
  logP = 0.0
  if x < pi.size:
    logP += np.log(pi[x])
    for i in xrange(0,pi.size):
      if i != x:
        logP += np.log(1.0-pi[i])
  return logP

def logBeta(x, alpha, beta):
  if type(x) is np.ndarray:
    N=x.size

    # at these x the value is only 1/log(B(alpha,beta))
    # because 0^0 = 1
    issue=(x<1.0e-15)|(1.0-x<1.0e-15) 

#    print('\t\talpha={}; beta={}'.format(alpha, beta))
#    print('\t\tbetaln={}'.format(-N*scisp.betaln(alpha,beta)))
#    print('\t\talphaterm={}'.format((alpha-1.0)*np.sum(np.log(x))))
#    print('\t\tbetaterm={}'.format((beta-1.0)*np.sum(np.log(1.0-x))))
#    print('\t\tbetaterm={}'.format(np.sum(np.log(1.0-x[~issue]))))
#    print('\t\tbetaterm={}'.format(np.log(1.0-x[~issue])))
#    print('\t\tbetaterm={}'.format(1.0-x))
#    print('\t\tbetaterm={}'.format(x))

  # CDF: - do not use!
  #print('\t\tterm={}'.format(np.log(scisp.betainc(alpha,beta,x))))
  #return np.sum(np.log(scisp.betainc(alpha,beta,x)))

    if alpha == 1.0:
      return -N*scisp.betaln(alpha,beta) \
          +(beta-1.0)*np.sum(np.log(1.0-x[~issue]))
    elif beta == 1.0:
      return -N*scisp.betaln(alpha,beta) \
          +(alpha-1.0)*np.sum(np.log(x[~issue]))
    else:
      return -N*scisp.betaln(alpha,beta) \
          +(alpha-1.0)*np.sum(np.log(x[~issue])) \
          +(beta-1.0)*np.sum(np.log(1.0-x[~issue]))
  else:
    if (x<1.0e-15)|(1.0-x<1.0e-15):
      return -scisp.betaln(alpha,beta)
    else:
      return -scisp.betaln(alpha,beta) \
          +(alpha-1.0)*np.log(x) \
          +(beta-1.0)*np.log(1.0-x)

def logDir(x, alpha):
  logP = 0.0
  if alpha.size == x.size:
    logP = scisp.gammaln(np.sum(alpha))
    for i in xrange(0, alpha.size):
      logP += -scisp.gammaln(alpha[i]) + (alpha[i]-1.0)*np.log(x[i])
  return logP


class HDP_sample:

  K=0; T=0; Nw=0
  # hyper parameters
  omega = 0; alpha = 0 
  Lambda = np.zeros(1)
  # parameters
  c = []
  z = []
  beta = np.zeros(1)
  v = np.zeros(1)
  sigV = np.zeros(1)
  pi = []
  sigPi = []
  # data
  x_tr = []
  x_ho = []
  # results
  perp = np.zeros(1)
  
  def __init__(self, K=None,T=None,Nw=None,omega=None,alpha=None,Lambda=None, pathToModel=None):
    if pathToModel is None:
      self.K = K
      self.T = T
      self.Nw = Nw
      self.omega = omega
      self.alpha = alpha
      self.Lambda = Lambda
    else:
      self.load(pathToModel)
  

  def save(s,path):
    print('len(x_tr)={0}'.format(len(s.x_tr)))
    sio.savemat(path,{'K':s.K,'T':s.T,'Nw':s.Nw,'omega':s.omega,'alpha':s.alpha,'Lambda':s.Lambda,'c':s.c,'z':s.z,'beta':s.beta,'v':s.v,'sigV':s.sigV,'pi':s.pi,'sigPi':s.sigPi,'x_tr':s.x_tr,'x_ho':s.x_ho,'perp':s.perp})

  def load(s,path):
    try:
      mat=sio.loadmat(path)
      print('Found model under {0}'.format(path))
    except Exception, err:
      print('Did not find model under {0}'.format(path))
      return False
    s.K=mat['K'][0][0]
    print('loaded K\t {0}'.format(s.K))
    s.T=mat['T'][0][0]
    print('loaded T\t {0}'.format(s.T))
    s.Nw=mat['Nw'][0][0]
    print('loaded Nw\t {0}'.format(s.Nw))
    s.omega=mat['omega'][0][0]
    print('loaded omega\t {0}'.format(s.omega))
    s.alpha=mat['alpha'][0][0]
    print('loaded alpha\t {0}'.format(s.alpha))
    s.Lambda=mat['Lambda']
    print('loaded Lambda\t {0}'.format(s.Lambda.shape))
    s.beta=mat['beta']
    print('loaded beta\t {0}'.format(s.beta.shape))
    s.sigV=mat['sigV']
    print('loaded sigV\t {0}'.format(s.sigV.shape))
    s.v=mat['v']
    print('loaded v\t {0}'.format(s.v.shape))
    s.perp = mat['perp']
    s.x_tr=[]; s.x_ho=[]; s.z=[]; s.c=[]; s.pi=[]; s.sigPi=[]
    for d in range(0,mat['x_ho'].shape[0]):
      s.x_ho.append(mat['x_ho'][d][0])

    for d in range(0,mat['x_tr'].shape[0]):
      s.x_tr.append(mat['x_tr'][d][0])
      #print('loaded x[{}]\t {}'.format(d,s.x[d].shape))
      s.z.append(mat['z'][d][0])
      #print('loaded z[{}]\t {}'.format(d,s.z[d].shape))
      s.c.append(mat['c'][d,:])
      #print('loaded c[{}]\t {}'.format(d,s.c[d].shape))
      s.pi.append(mat['pi'][d,:])
      #print('loaded pi[{}]\t {}'.format(d,s.pi[d].shape))
      s.sigPi.append(mat['sigPi'][d,:])
      #print('loaded sigPi[{}]\t {}'.format(d,s.sigPi[d].shape))
    return True


  def loadHDPSample(self, x_tr, x_ho, hdp):
    self.x_tr = x_tr
    self.x_ho = x_ho
    if isinstance(hdp,bnp.HDP_var) or isinstance(hdp,bnp.HDP_var_ss):
      D=len(self.x_tr)
      D_ho=len(self.x_ho)
      print("---------------------- Loading from hdp -------------------------");
      self.sigV = np.zeros(self.K+1,dtype=np.double)
      self.v = np.zeros(self.K,dtype=np.double)
      hdp.getCorpTopicProportions(self.v,self.sigV)
      #print('topic proportions: \t{}\t{}'.format(self.sigV,np.sum(self.sigV)))
      #print('GT topic proportions: \t{}\t{}'.format(gtCorpProp,np.sum(gtCorpProp)))
      #print("---------------------- Corpus Topics -------------------------");
      self.beta=[]
      for k in range(0,self.K):
        self.beta.append(np.zeros(self.Nw,dtype=np.double))
        hdp.getCorpTopic(self.beta[k],k)
        #print('self.beta_{}=\t{}'.format(k,topic[k]))
        #print('gtTopic_{}=\t{}'.format(k,gtTopic[k,:]))
      self.sigPi=[]
      self.pi=[]
      self.c=[]
      self.z=[] # word indices to doc topics
      for d in range(0,D):
        self.sigPi.append(np.zeros(self.T+1,dtype=np.double))
        self.pi.append(np.zeros(self.T,dtype=np.double))
        self.c.append(np.zeros(self.T,dtype=np.uint32))
        print('getting {}'.format(d))
        hdp.getDocTopics(self.pi[d],self.sigPi[d],self.c[d],d)
        print('pi({0}): {1}'.format(d,self.pi[d]))
        print('sigPi({0}): {1}'.format(d,self.sigPi[d]))
        #print('c({0}): {1}'.format(d,self.c[d]))
        self.z.append(np.zeros(self.x_tr[d].size,dtype=np.uint32))
        hdp.getWordTopics(self.z[d],d)
        print('word topics ({}) size: {}'.format(d,self.z[d].shape))
      self.perp = np.zeros(D)
      hdp.getPerplexity(self.perp)
      print('Perplexity of iterations: {}'.format(self.perp))
    else:
      print('Error loading hdp of type {}'.format(type(hdp)))

  def checkSticks(self):
    print('--------------------- Checking Stick pieces -----------------')
    print('sigV = {0}; {1}'.format(self.sigV,np.sum(self.sigV)))
    D=len(self.x_tr)
    for d in range(0,D):
      np.sum(self.sigPi[d])
      print('sigPi = {0}; {1}'.format(self.sigPi[d],np.sum(self.sigPi[d])))

  def KLdivergence(self,q):
    kl = 0.0
    logP_joint = 0.0
    logQ_joint = 0.0
    D=len(self.x_tr)
    for d in range(0,D):
      N=self.x_tr[d].size
      for n in range(0,N):
        logP = self.logP_wordJoint(d,n)
        logQ = q.logP_wordJoint(d,n)
        logP_joint += logP
        logQ_joint += logQ
        kl += (logP - logQ)* np.exp(logP)
    return kl, logP_joint, logQ_joint

  # x is the heldout data i.e. a document from the same dataset as was trained on
  #def Perplexity(self,x):

  def CrossEntropy(self,x):
    H=0
    N = x.size
    for w in range(0,self.Nw):
      n_w = np.sum(x==w)
      q = 0.5 # TODO: computation of q given x 
      H -= (n_w/N) * np.log(q)

  def docTopicsImg(self):
    D = len(self.x_tr)
    # create image for topic
    vT = np.zeros((self.K,D))
    for d in range(0,D):
      for t in range(0,self.T):
        #print('{0} {1} c.shape={2}'.format(d,t,self.c[d].shape))
        k=self.c[d][t]
        vT[k,d] += self.sigPi[d][t]
#    print('vT={0}'.format(vT))
#    print('vT_norm={0}'.format(np.sum(vT,0)))
#    print('sigPi_d={0}'.format(self.sigPi[d]))
    return vT

  def plotTopics(self,minSupport=None):
    D = len(self.x_tr)
    ks=np.zeros(D)
    for d in range(0,D):
      t_max=np.nonzero(self.sigPi[d]==np.max(self.sigPi[d]))[0][0]
      print('d={}; D={}'.format(d,D))
      print('t_max={};'.format(np.nonzero(self.sigPi[d]==np.max(self.sigPi[d]))))
      print('sigPi={}; sum={}'.format(self.sigPi[d],np.sum(self.sigPi[d])))
      print('c[{}]={};'.format(d,self.c[d]))
      if t_max < self.c[d].size:
        k_max = self.c[d][t_max]
      else:
        k_max = np.nan # this means that we arre not selecting one of the estimated models!! (the last element in sigPi is 1-sum(sigPi(0:end-1)) and represents the "other" models
      ks[d]=k_max
    ks_unique=np.unique(ks)
    ks_unique=ks_unique[~np.isnan(ks_unique)]
    if minSupport is not None:
      Np = ks_unique.size # numer of subplots
      print('D{0} Np{1}'.format(D,Np))
      sup = np.zeros(ks_unique.size)
      for d in range(0,D):
        sup[np.nonzero(ks_unique==ks[d])[0]] += 1
      print('sup={0} sum(sup)={1}'.format(sup,np.sum(sup)))
      delete = np.zeros(ks_unique.size,dtype=np.bool)
      for i in range(0,Np):
        if sup[i] < minSupport:
          delete[i]=True
      ks_unique = ks_unique[~delete] 
    Np = ks_unique.size # numer of subplots
    print('D{0} Np{1}'.format(D,Np))
    Nrow = np.ceil(np.sqrt(Np))
    Ncol = np.ceil(np.sqrt(Np))
    fig=plt.figure()
    for i in range(0,Np):
      plt.subplot(Ncol,Nrow,i+1)
      x = np.linspace(0,self.beta[int(ks_unique[i])].size-1,self.beta[int(ks_unique[i])].size)
      plt.stem(x,self.beta[int(ks_unique[i])])
      plt.xlabel('topic '+str(ks_unique[i]))
    return fig

  def generateDirHDPSample(self,D,N):
    # doc level
    self.x_tr=[]
    # draw K topics from Dirichlet
    self.beta = np.random.dirichlet(self.Lambda,self.K)
    # draw breaking proportions using Beta
    self.v = np.random.beta(1,self.omega,self.K-1)
    self.sigV = stickBreaking(self.v)
  
    self.c=[]  # pointers to corp level topics
    self.pi=[] # breaking proportions for selected doc level topics
    self.sigPi=[]
    self.z=[]
  
    for d in range(0,D): # for each document
      # draw T doc level pointers to topics (multinomial)
      self.c.append(np.zeros(self.T))
      _, self.c[d] = np.nonzero(np.random.multinomial(1,self.sigV,self.T))
      # draw T doc level breaking proportions using Beta
      self.pi.append(np.zeros(self.T-1))
      self.sigPi.append(np.zeros(self.T))
      self.pi[d] = np.random.beta(1,self.alpha,self.T-1)
      self.sigPi[d] = stickBreaking(self.pi[d])
  
      self.x_tr.append(np.zeros(N))
      # draw topic assignment of word (multinomial)
      self.z.append(np.zeros(N))
      _, self.z[d] = np.nonzero(np.random.multinomial(1,self.sigPi[d],N))
      for i in range(0,N): # for each word
        # draw words
        _, self.x_tr[d][i] = np.nonzero(np.random.multinomial(1,self.beta[ self.c[d][ self.z[d][i]], :],1))
  
      self.x_tr[d] = self.x_tr[d].astype(np.uint32)
  
#    for d in range(0,D):
#      print('d={0}: {1}'.format(d,self.x_tr[d]))
  
    return self.x_tr, self.sigV, self.beta, self.pi, self.c

  
  def logP_fullJoint(self):
    logP = 0.0
    D = len(self.x_tr)
    for d in range(0,D):
      N = self.x_tr[d].size
      for n in range(0,N):
        logP_w = self.logP_wordJoint(d,n)
#        print('logP({},{})={}'.format(d,n,logP_w))
        logP += logP_w
    return logP

  def logP_wordJoint(self, d, n):
    #print('d={}, n={}'.format(d,n))
    #print('x={}; x.len={}; x[d].size={}; c.len={}; z.len={}; v.size={}; sigV.size={}; pi.len={}; sigPi.len={}'.format(self.x[d][n],len(self.x),self.x[d].size,len(self.c),len(self.z),self.v.size,self.sigV.size,len(self.pi),len(self.sigPi)))
    #print('z={}; c={}; beta.size {}\n'.format(self.z[d][n], self.c[d][ self.z[d][n]],self.beta.shape))

#    print('\tx|beta =    {}'.format(logCat(self.x[d][n], self.beta[ self.c[d][ self.z[d][n]]])))
#    print('\tc|sigV =    {}'.format(logCat(self.c[d][ self.z[d][n]], self.sigV)))
#    print('\tv|omega =   {}'.format(logBeta(self.v, 1.0, self.omega)))
#    print('\tz|sigPi =   {}'.format(logCat(self.z[d][n], self.sigPi[d])))
#    print('\tpi|alpha =  {}'.format(logBeta(self.pi[d], 1.0, self.alpha)))
#    print('\tbeta|lambda={}'.format(logDir(self.beta[ self.c[d][ self.z[d][n]]], self.Lambda)))
    return logCat(self.x_tr[d][n], self.beta[ self.c[d][ self.z[d][n]]]) \
    + logCat(self.c[d][ self.z[d][n]], self.sigV) \
    + logBeta(self.v, 1.0, self.omega) \
    + logCat(self.z[d][n], self.sigPi[d]) \
    + logBeta(self.pi[d], 1.0, self.alpha) \
    + logDir(self.beta[ self.c[d][ self.z[d][n]]], self.Lambda)
