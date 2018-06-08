# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:22:39 2018
Question 1 (9.5/10)

Your Hermite polynomial needs to raise an exception for n < 0.


@author: Jason (U1440158A)
"""
#%%
### Question 1
## Part a
from scipy import arange, linspace
import matplotlib.pyplot as plt
xmin, xmax, nx = -4, 4, 250

n = arange(0,4) # or range(0,4)
x = linspace(xmin, xmax, nx)

def hermitePol(n,x):
    #######
    # Can be removed if required package loaded earlier
    # Ensure function is usable even the package not called globally
    #######
    from scipy import array
    if (type(x) == list) | (type(x) == range): # Convert list or range datatype into array for vectorization
        print('Convert object %s into array' % type(x), end = '\n')
        x = array(x)
    if n == 0:
        return 1+0*x
    elif n == 1:
        return 2*x
    else:
        return 2*x*hermitePol(n-1,x) - 2*(n-1)*hermitePol(n-2,x)
    
def harmonicOscillatorFunc(n,x):
    #######
    # Can be removed if required package loaded earlier
    # Ensure function is usable even the package not called globally
    from scipy import array, exp, sqrt, pi
    from math import factorial
    #######
    if (type(x) == list) | (type(x) == range): # Convert list or range datatype into array for vectorization
        print('Convert object %s into array' % type(x), end = '\n')
        x = array(x)
    return exp(-x**2 / 2) * hermitePol(n,x) * 1 / (sqrt(2**n * factorial(n) * sqrt(pi)))

print('### Question 1 \n ## Part a')
fig = plt.figure(figsize=(15,10))
for valueN in n:
    # Loop for different state value e.g ground state, 1st state, etc.
    # Plot the corresponding wavefunction and put the label for respective n
    y = harmonicOscillatorFunc(valueN, x)
    plt.plot(x,y, label = '1D Quantum Harmonic Oscillator Wave Function n = %d' % valueN)

plt.xlabel('x')
plt.ylabel('psi(x)')
plt.title('1D Quantum Harmonic Oscillator Wave Function for different state, psi_n')
plt.legend()
plt.show()

#%%
### Question 1
## Part b
#import time
#start = time.time()

print('### Question 1 \n ## Part b')
fig1 = plt.figure(figsize=(15,10))
nPartB = 30
xminB, xmaxB, nxB = -10, 10, 100
xB = linspace(xminB, xmaxB, nxB)
plt.plot(xB, harmonicOscillatorFunc(nPartB,xB))
plt.xlabel('x')
plt.ylabel('psi(x)')
plt.title('1D Quantum Harmonic Oscillator Wave Function for n = %d' % nPartB)
#plt.legend()
plt.show()

#end = time.time()
#print(end - start)

#print('###Time required to run is %.4f seconds' % (end-start))
# ~12 seconds run

#%%
### Question 1
## Part c
nPartC = 5
import scipy.integrate as integrate
#def integrand(x):
#    return x**2 * harmonicOscillatorFunc(n,x)**2

from scipy import inf, sqrt

result = integrate.quad(lambda x : x**2 * harmonicOscillatorFunc(nPartC,x)**2, -inf, inf, limit = 100)

print('### Question 1 \n ## Part c')
print('The quantum position uncertainty value for n = %d is %.4f' % (nPartC, sqrt(result[0])))

