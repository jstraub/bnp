
import numpy as np
import libbnp as bnp

def generateDirHDPSample(D,N,K,T):

  x=[]
  # draw K topics from Dirichlet
  
  # draw breaking proportions using Beta

  for d in range(0,D): # for each document
    # draw T doc level pointers to topics (multinomial)
    
    # draw T doc level breaking proportions using Beta

    for i in range(0,N): # for each word
      # draw topic assignment of word (multinomial)

      # draw words


  return x

