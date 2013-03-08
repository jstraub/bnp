
import numpy as np
import libbnp as bnp
import scipy.special as scisp
import pdb

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

  #print('\t\tpi = {}'.format(pi))
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

  K=T=Nw=0
  # hyper parameters
  omega = alpha = 0 
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
  x = []
  
  def __init__(self, K,T,Nw,omega,alpha,Lambda):
    self.K = K
    self.T = T
    self.Nw = Nw
    self.omega = omega
    self.alpha = alpha
    self.Lambda = Lambda

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
    return logCat(self.x[d][n], self.beta[ self.c[d][ self.z[d][n]]]) \
    + logCat(self.c[d][ self.z[d][n]], self.sigV) \
    + logBeta(self.v, 1.0, self.omega) \
    + logCat(self.z[d][n], self.sigPi[d]) \
    + logBeta(self.pi[d], 1.0, self.alpha) \
    + logDir(self.beta[ self.c[d][ self.z[d][n]]], self.Lambda)

  def logP_fullJoint(self):
    logP = 0.0
    D = len(self.x)
    for d in range(0,D):
      N = self.x[d].size
      for n in range(0,N):
        logP_w = self.logP_wordJoint(d,n)
#        print('logP({},{})={}'.format(d,n,logP_w))
        logP += logP_w
    return logP

  def loadHDPSample(self,x,beta=None,c=None,z=None,v=None,sigV=None,pi=None,sigPi=None,omega=None,alpha=None,Lambda=None, hdp=None):
    self.x = x
    if hdp is None:
      self.beta = beta
      self.c = c
      self.z = z
      self.v = v
      self.sigV = sigV
      self.pi = pi
      self.sigPi = sigPi
      self.omega = omega
      self.alpha = alpha
      self.Lambda = Lambda
    else:
      D=len(self.x)
      #print("---------------------- Corpus Topic Proportions -------------------------");
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
        hdp.getDocTopics(self.pi[d],self.sigPi[d],self.c[d],d)
        print('pi({}): {}'.format(d,self.pi[d]))
        print('sigPi({}): {}'.format(d,self.sigPi[d]))
        #print('c({}): {}'.format(d,self.c[d]))
        self.z.append(np.zeros(self.x[d].size,dtype=np.uint32))
        hdp.getWordTopics(self.z[d],d)

  def checkSticks(self):
    print('sigV = {}; {}'.format(self.sigV,np.sum(self.sigV)))
    D=len(self.x)
    for d in range(0,D):
      np.sum(self.sigPi[d])
      print('sigPi = {}; {}'.format(self.sigPi[d],np.sum(self.sigPi[d])))
      
  def KLdivergence(self,q):
    kl = .0;
    D=len(self.x)
    for d in range(0,D):
      N=self.x[d].size
      for n in range(0,N):
        kl += (self.logP_wordJoint(d,n) - q.logP_wordJoint(d,n))* np.exp(self.logP_wordJoint(d,n))
    return kl


  def docTopicsImg(self):
    D = len(self.x)
    # create image for topic
    vT = np.zeros((self.K,D))
    for d in range(0,D):
      for t in range(0,self.T):
        k=self.c[d][t]
        vT[k,d]=self.sigPi[d][t]
    return vT

  def generateDirHDPSample(self,D,N):
    
    # doc level
    self.x=[]
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
  
      self.x.append(np.zeros(N))
      # draw topic assignment of word (multinomial)
      self.z.append(np.zeros(N))
      _, self.z[d] = np.nonzero(np.random.multinomial(1,self.sigPi[d],N))
      for i in range(0,N): # for each word
        # draw words
        _, self.x[d][i] = np.nonzero(np.random.multinomial(1,self.beta[ self.c[d][ self.z[d][i]], :],1))
  
      self.x[d] = self.x[d].astype(np.uint32)
  
#    for d in range(0,D):
#      print('d={}: {}'.format(d,self.x[d]))
  
    return self.x, self.sigV, self.beta, self.pi, self.c
  
