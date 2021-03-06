
import numpy as np
import libbnp as bnp
import scipy.special as scisp
import scipy.io as sio

import matplotlib.pyplot as plt
import pickle

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


class HDP_base:

#  # parameters
#  c = []
#  z = []
#  beta = np.zeros(1)
#  v = np.zeros(1)
#  sigV = np.zeros(1)
#  pi = []
#  sigPi = []
#  # data
#  x_tr = []
#  x_ho = []
#  # results
#  perp = np.zeros(1)

  loaded = dict()
  
  def __init__(s, K=None,T=None,Nw=None,omega=None,alpha=None,base=None, pathToModel=None):

    s.scalars = ['K','T','Nw','omega','alpha','D_tr','D_te']
    s.matrices = ['c','pi','sigPi','base','beta','sigV','v','perp']
    s.listMatrices=['z','x_tr','x_te','logP_w']
    
    s.shape=dict()

#    s.states=[]
#    s.states.expand(s.scalars)
#    s.states.expand(s.matrices)
#    s.states.expand(s.listMatrices)

    s.state=dict()
    if pathToModel is None:
      s.state['K'] = K
      s.state['T'] = T
      s.state['Nw'] = Nw
      # hyper parameters
      s.state['omega'] = omega
      s.state['alpha'] = alpha
      s.state['base'] = np.zeros(base.rowDim())
      base.asRow(s.state['base'])
      print('in HDP_base setting K={}; T={}; Nw={}; omega={}; alpha={};'.format(s.state['K'],s.state['T'],s.state['Nw'],s.state['omega'],s.state['alpha']))
    else:
      s.load(pathToModel)

  def save(s,path):
    pickle.dump(s.state,open(path,'w'))
#    sio.savemat(path,s.state,oned_as='row')

  def load(s,path):
    try:
#      s.loaded=sio.loadmat(path)
      s.state=pickle.load(open(path,'r'))
      print('Found model under {0}'.format(path))
    except Exception, err:
      print('Did not find model under {0}'.format(path))
      return False
#    print('--- loading state from mat file at {}'.format(path))
#    if HDP_base.__parseLoad(s,s.loaded):
#      print('--- loaded state successfully!')
#    else:
#      print('--- error while loading state!')
    return True

  def __parseLoad(s,mat):
    print('mat.keys: {}'.format(mat.keys()))

    for scalar in s.scalars:
      s.state[scalar]=mat[scalar][0][0]
      print('loaded {}\t {}'.format(scalar,s.state[scalar]))
    for matrix in s.matrices:
      s.state[matrix]=mat[matrix]
      if s.state[matrix].shape[1] == 1: # savemat/loadmat puts vectors always as column vectors
        s.state[matrix] = s.state[matrix].ravel()
      print('loaded {}\t {}'.format(matrix,s.state[matrix].shape))

# have to be added to the appropriate matrices or listMatrices
#    s.state['x_tr'] = mat['x_tr']
#    print('loaded x_tr\t {}'.format(s.state['x_tr'].shape))
#    s.state['x_ho'] = mat['x_ho']

    for listMatrix in s.listMatrices:
      s.state[listMatrix] = []
      D=s.state['D_tr'] #mat[listMatrix].size
      if listMatrix == 'x_te':
        D=s.state['D_te']
      print('{} D={}'.format(listMatrix,D))
      if D==1:
        s.state[listMatrix].append(mat[listMatrix].ravel())
        #print('loaded {}_{}\t {}'.format(listMatrix,d,s.state[listMatrix][0].shape))
      else:
        for d in range(0,D):
          s.state[listMatrix].append(mat[listMatrix][d][0])
          if len(s.state[listMatrix][d].shape) >1:
            if s.state[listMatrix][d].shape[1] == 1: # savemat/loadmat puts vectors always as column vectors
              s.state[listMatrix][d] = s.state[listMatrix][d].ravel()
          #print('loaded {}_{}\t {}'.format(listMatrix,d,s.state[listMatrix][d].shape))
      print('loaded {}\t {}'.format(listMatrix,D))
    return True


  def stateEquals(s,hdp):
    print('--- checking whether state of two hdps is equal')
    for key, val in s.state.iteritems():
      print('checking {}'.format(key))
      if key in s.scalars:
        if hdp.state[key] != val:
          print('keys {} differ'.format(key))
          return False
      elif key in s.matrices:
        if np.any(hdp.state[key] != val):
          print('keys {} differ'.format(key))
          print('{}\nvs\n{}'.format(val,hdp.state[key]))
          return False
      elif key in s.listMatrices:
        D=len(val)
        for d in range(0,D):
          if np.any(hdp.state[key][d] != val[d]):
            print('keys {} at d={} differ'.format(key,d))
            print('{}\nvs\n{}'.format(val[d],hdp.state[key][d]))
            return False
    print('--- HDPs state is equal!')
    return True


  def loadHDPSample(s, x_tr, x_te, hdp):
    if isinstance(hdp,bnp.HDP_var_Dir) or isinstance(hdp,bnp.HDP_var_NIW) or isinstance(hdp,bnp.HDP_var_ss):
      print("---------------------- obtaining results -------------------------");
      s.state['x_tr'] = x_tr
      s.state['x_te'] = x_te
      #print('{}'.format(s.state['x_tr']))

      D_tr=s.state['D_tr']=len(s.state['x_tr'])
      D_te=s.state['D_te']=len(s.state['x_te'])
      print('D_tr={}; D_te={};'.format(s.state['D_tr'],s.state['D_te']))

      s.state['sigV'] = np.zeros(s.state['K']+1,dtype=np.double)
      s.state['v'] = np.zeros(s.state['K'],dtype=np.double)
      hdp.getCorpTopicProportions(s.state['v'],s.state['sigV'])
      print('gotCorpTopicProportions {} {}'.format(s.state['v'].shape,s.state['sigV'].shape))
      
      s.state['perp'] = np.zeros(D_tr)
      hdp.getPerplexity(s.state['perp'])
      print('Perplexity of iterations: {}'.format(s.state['perp']))

      betaCols = hdp.getTopicsDescriptionLength()
      s.state['beta']= np.zeros((s.state['K'],betaCols),dtype=np.double)
      hdp.getCorpTopics(s.state['beta'])
      print('beta={}'.format(s.state['beta'].shape))
      print('beta={}'.format(s.state['beta']))

      s.state['sigPi']=np.zeros((D_tr,s.state['T']+1),dtype=np.double)
      s.state['pi']=np.zeros((D_tr,s.state['T']),dtype=np.double)
      s.state['c']=np.zeros((D_tr,s.state['T']),dtype=np.uint32)
      if hdp.getDocTopics(s.state['pi'],s.state['sigPi'],s.state['c']):
        print('pi: {}'.format(s.state['pi'].shape))
        #print('pi: {}'.format(s.state['pi']))
        print('sigPi: {}'.format(s.state['sigPi'].shape))
        #print('sigPi: {}'.format(s.state['sigPi']))
      else:
        print('error while loading pi, sigPi and c from C++ model')

      s.state['logP_w'] = [] #np.zeros((D_tr,s.state['Nw']),dtype=np.double)
      s.state['z']=[] # word indices to doc topics
      for d in range(0,D_tr):
        N_d = s.state['x_tr'][d].size
        #print('N_d={}'.format(N_d))

#        print('getting {}'.format(d))
#        hdp.getDocTopics(s.state['pi'][d,:],s.state['sigPi'][d,:],s.state['c'][d,:],d)
        #print('c({0}): {1}'.format(d,s.state['c'][d]))
        s.state['z'].append(np.zeros(N_d,dtype=np.uint32))
        hdp.getWordTopics(s.state['z'][d],d)
        s.state['logP_w'].append(np.zeros(N_d,dtype=np.double))
        hdp.getWordDistr(s.state['logP_w'][d],d)
        
    else:
      print('Error loading hdp of type {}'.format(type(hdp)))

  def checkSticks(s):
    print('--------------------- Checking Stick pieces -----------------')
    print('sigV = {0}; {1}'.format(s.state['sigV'],np.sum(self.sigV)))
    D=len(s.state['x_tr'])
    for d in range(0,D):
      np.sum(s.state['sigPi'][d])
      print('sigPi = {0}; {1}'.format(s.state['sigPi'][d],np.sum(self.sigPi[d])))

# KL divergence

  def klD(s,logP,logQ):
    p = np.exp(logP)
    if len(logP.shape) > 1:
      kl = np.zeros(p.shape[0])
      for i in range(0,logP.shape[0]):
        kl[i] = np.sum(p[i,:]*(logP[i,:]-logQ[i,:]))
      return kl
    else:
      return np.sum(p*(logP-logQ))

#  symmeterised divergence
  def symKL(s,logP,logQ):
    return np.sum((logP-logQ)*(np.exp(logP)-np.exp(logQ)))

  def symKLImg(s):
    D_tr = s.state['D_tr']
    K = s.state['K']
    T = s.state['T']
    # create image for topic
    symKLd = np.zeros((D_tr,D_tr))
    logP = s.state['logP_w']
    p = np.exp(logP)
    for di in range(0,D_tr):
      for dj in range(0,D_tr):
        symKLd[di,dj] =  np.sum((logP[di,:]-logP[dj,:])*(p[di,:]-p[dj,:]))
#s.symKL(s.state['logP_w'][di],s.state['logP_w'][dj])
    return symKLd

# Jensen-Shannon Divergence - 
  def jsD(s,logP,logQ):
    p=np.exp(logP)
    q=np.exp(logQ)
    logM=-np.log(2)+np.log(p + q)
    return 0.5*np.sum(p*(logP-logM))+0.5*np.sum(q*(logQ-logM))
  def jsDImg(s):
    D_tr = s.state['D_tr']
    K = s.state['K']
    T = s.state['T']
    # create image for topic
    jsd = np.zeros((D_tr,D_tr))
    for di in range(0,D_tr):
      for dj in range(0,D_tr):
        jsd[di,dj] = s.jsD(s.state['logP_w'][di],s.state['logP_w'][dj])
    return jsd

  def docTopicsImg(s):
    D_tr = s.state['D_tr']
    K = s.state['K']
    T = s.state['T']
    # create image for topic
    vT = np.zeros((K,D_tr))
    for d in range(0,D_tr):
      for t in range(0,T):
        #print('{0} {1} c.shape={2}'.format(d,t,s.c[d].shape))
        k=s.state['c'][d][t]
        vT[k,d] += s.state['sigPi'][d][t]
#    print('vT={0}'.format(vT))
#    print('vT_norm={0}'.format(np.sum(vT,0)))
#    print('sigPi_d={0}'.format(s.sigPi[d]))
    return vT

  def plotTopics(s,minSupport=None):
    D = s.state['D_tr']
    ks=np.zeros(D)
    for d in range(0,D):

      # necessary since topics may be selected several times!
      c_u=np.unique(s.state['c'][d,:])
      sigPi_u = np.zeros(c_u.size)
      for i in range(0,c_u.size):
        #print('{}'.format(c_u[i] == s.c[d]))
        #print('{}'.format(s.sigPi[d]))
        sigPi_u[i] = np.sum(s.state['sigPi'][d,c_u[i] == s.state['c'][d,:]])
      k_max = c_u[sigPi_u == np.max(sigPi_u)]
#      print('c={};'.format(s.c[d]))
#      print('sigPi={};'.format(s.sigPi[d]))
      print('sigPi_u = {};\tc_u={};\tk_max={}'.format(sigPi_u,c_u,k_max))

#      t_max=np.nonzero(s.sigPi[d]==np.max(s.sigPi[d]))[0][0]
#      print('d={}; D={}'.format(d,D))
#      print('t_max={};'.format(np.nonzero(s.sigPi[d]==np.max(s.sigPi[d]))))
#      print('sigPi={}; sum={}'.format(s.sigPi[d],np.sum(s.sigPi[d])))
#      print('c[{}]={};'.format(d,s.c[d]))
#      if t_max < s.c[d].size:
#        k_max = s.c[d][t_max]
#      else:
#        k_max = np.nan # this means that we arre not selecting one of the estimated models!! (the last element in sigPi is 1-sum(sigPi(0:end-1)) and represents the "other" models
      ks[d]=k_max[0]
    ks_unique=np.unique(ks)
    ks_unique=ks_unique[~np.isnan(ks_unique)]
    if minSupport is not None:
      Np = ks_unique.size # numer of subplots
      #print('D{0} Np{1}'.format(D,Np))
      sup = np.zeros(ks_unique.size)
      for d in range(0,D):
        sup[np.nonzero(ks_unique==ks[d])[0]] += 1
      #print('sup={0} sum(sup)={1}'.format(sup,np.sum(sup)))
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
      x = np.linspace(0,s.state['beta'][int(ks_unique[i])].size-1,s.state['beta'][int(ks_unique[i])].size)
      plt.stem(x,s.state['beta'][int(ks_unique[i])])
      plt.ylim([0.0,1.0])
      plt.xlabel('topic '+str(ks_unique[i]))
    return fig


class HDP_sample(HDP_base):

  def generateDirHDPSample(s,D,N):
    # doc level
    s.state['D_tr'] = D
    s.state['x_tr']=[]
    # draw K topics from Dirichlet
    s.state['beta'] = np.random.dirichlet(s.state['base'],s.state['K'])
    # draw breaking proportions using Beta
    s.state['v'] = np.random.beta(1,s.state['omega'],s.state['K'])
    s.state['sigV'] = stickBreaking(s.state['v'])
  
    s.state['c']=[]  # pointers to corp level topics
    s.state['pi']=[] # breaking proportions for selected doc level topics
    s.state['sigPi']=[]
    s.state['z']=[]


    s.state['c'] = np.zeros((D,s.state['T']))
    s.state['pi'] = np.zeros((D,s.state['T']))
    s.state['sigPi'] = np.zeros((D,s.state['T']+1))
  
    s.state['logP_w']=np.ones((D,s.state['Nw']))*0.1
    for d in range(0,D): # for each document
      # draw T doc level pointers to topics (multinomial)
      _, s.state['c'][d,:] = np.nonzero(np.random.multinomial(1,s.state['sigV'],s.state['T']))
      # draw T doc level breaking proportions using Beta
      s.state['pi'][d,:] = np.random.beta(1,s.state['alpha'],s.state['T'])
      s.state['sigPi'][d,:] = stickBreaking(s.state['pi'][d,:])
  
      s.state['x_tr'].append(np.zeros(N))
      # draw topic assignment of word (multinomial)
      s.state['z'].append(np.zeros(N))
      _, s.state['z'][d] = np.nonzero(np.random.multinomial(1,s.state['sigPi'][d,:],N))
      for i in range(0,N): # for each word
        # draw words
        _, s.state['x_tr'][d][i] = np.nonzero(np.random.multinomial(1,s.state['beta'][ s.state['c'][d, s.state['z'][d][i]], :],1))
        s.state['logP_w'][d, s.state['x_tr'][d][i]] +=1
  
      s.state['x_tr'][d] = s.state['x_tr'][d].astype(np.uint32)
      s.state['logP_w'][d,:] /= np.sum(s.state['logP_w'][d,:])
    s.state['logP_w'] = np.log(s.state['logP_w'])
  
#    for d in range(0,D):
#      print('d={0}: {1}'.format(d,s.state['x_tr'][d]))
    s.state['D_te'] = 0
    s.state['perp'] = []
  
    return s.state['x_tr'], s.state['sigV'], s.state['beta'], s.state['pi'], s.state['c']

  
    
    
    




