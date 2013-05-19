import numpy as np

a=np.random.normal(0,2,(100,100))
f=open('testNormal.txt','w')
for i in range(0,a.shape[0]):
  for j in range(0,a.shape[1]):
    f.write('{} '.format(a[i,j]))
  f.write('\n')
f.close()
