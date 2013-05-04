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
  book=dict()
  i=0
  for d in dicts:
    f=open(d)
    for line in f:
      book[line.strip()] = i
      i+=1
    f.close()
  print('book.size={}'.format(len(book)))

  stopWords=set(['the','of','to','for','is','are','on'])

  for line in fileinput.input():
    words=re.split('[ .,;:]',line)
    quant=[]
    for word in words:
      if word not in stopWords:
        if word in book:
          print('word: {} = {}'.format(word,book[word]))
          quant.append(str( book[word] ))
    print(' '.join(quant))




