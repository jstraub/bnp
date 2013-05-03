#!/usr/bin/python

# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or 
# http://www.opensource.org/licenses/mit-license.php 

import numpy as np

import time
import cProfile
import argparse
import re

import fileinput

if __name__ == '__main__':

  dicts = ['../data/ispell/english.0','../data/ispell/english.1','../data/ispell/english.2','../data/ispell/english.3']
  book=[]
  for d in dicts:
    f=open(d)
    for line in f:
      book.append(line.strip())
    f.close()
  #print('book.size={}'.format(len(book)))
  #print(' '.join(book))

  for line in fileinput.input():
    words=re.split('[ .,;:]',line)
    quant=[]
    for word in words:
      nr = 0
      #print('checking {}'.format(word))
      for bookw in book:
        if word == bookw:
          quant.append(str(nr))
          #print('{} = #{}'.format(word,nr))
          break
        nr += 1
    print(' '.join(quant))




