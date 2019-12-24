import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def function(w):
    return 3 * w * w + 2 * w + 1

w = np.arange(-2,2,0.01)

y = function(w)

plt.plot(w,y,'r-')

alpha = 0.1 # learning rate
nb_max_iter = 100
eps=0.0001 # stop condition

w0=1.5
y0=function(w0)
plt.scatter(w0,y0)
cond=eps+10.0
nb_iter=0
tmp_y=y0

while cond> eps and nb_iter<nb_max_iter:
    w0=w0-alpha*misc.derivative(function,w0)
    y0=function(w0)
    nb_iter=nb_iter+1
    cond=abs(tmp_y-y0)
    tmp_y=y0
    print(w0,y0,cond)
    plt.scatter(w0,y0)
    
plt.title('Gradrient Descent')
plt.grid()
plt.savefig('gradient.png',bbox_inches='tight')
plt.show()