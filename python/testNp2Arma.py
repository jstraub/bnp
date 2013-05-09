
import libbnp as bnp
import numpy as np

A=np.zeros((3,3))
Arr=np.zeros((1,3))
Ar=np.zeros(3)
Acc=np.zeros((3,1))
Ac=np.zeros(3)

t=bnp.TestNp2Arma()

print '---------'
t.getAmat(A)
print('{}'.format(A))
print '---------'
t.getArow(Arr)
print('{}'.format(Arr))
print '---------'
t.getArow(Ar)
print('{}'.format(Ar))
print '---------'

t.getAcol(Acc)
print('{}'.format(Acc))
print '---------'
t.getAcol(Ac)
print('{}'.format(Ac))
print '---------'
