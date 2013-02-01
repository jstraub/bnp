#! /usr/bin/env python


import numpy as np
import libhdp as bnp

A=np.zeros((4,4), dtype=np.uint32)
Aa=np.zeros(34)


Dir=bnp.Dir(Aa)

bnp.HDP_Dir(Dir,10.0,10.0)
