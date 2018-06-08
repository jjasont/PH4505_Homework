# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:22:39 2018

@author: Jason Tanuwijaya
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

#%% Question 2
## Part a
from scipy import linspace
import scipy.integrate as integrate
import matplotlib.pyplot as plt

mass = 10.0*1E3 #conversion from metric ton to kilogram
G = 6.674E-11
sideLength = 10.0 #in meter
rho = mass/(sideLength**2)

def Fz(Z):
    return G * rho * Z *integrate.dblquad(lambda x, y: 1/(x**2 + y**2 + Z**2)**(3/2), -sideLength/2.0, sideLength/2.0, lambda x: -sideLength/2.0, lambda x: sideLength/2.0)[0]
    
# Initalize the value of z from 0 to 10
zmin, zmax, nz = 0, 10, 100
Z_array = linspace(zmin, zmax, nz)
fig2 = plt.figure(figsize = (10,10))
ForceZ = [Fz(itemZ) for itemZ in Z_array] # List comprehension for the Force in z-direction calculation
# ForceZ contain the value for the forze at Z_array entries distance
plt.plot(Z_array, ForceZ)
plt.xlabel('Distance of the plate to the point (z)')
plt.ylabel('Force (Newton)')
plt.title('Force along the z-axis for 10m by 10m plate weight 10 metric tonne')
plt.show()

#%% Question 3
## Part a & b
#from scipy import *
#import numpy as np
#import random as random
#import math as math
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#
#def volume_torus(N = 3500, r = 0.5, R = 1.0):
#    assert R >= 0
#    assert r >= 0
#    assert N > 0
#    #Create 2 arrays. 1 to store the values we want and the other is to
#    #store the discarded values
#    vol = []
#    out_vol = []
#    outer_mostR = R+r
#    #inner_mostR = R-r
#    
#    for i in range(N):        
#        # Generate a random value between -outer_mostR to outer_mostR for x, y, and z
#        # Shell of observation is cube with sides 2*outer_mostR
#        x = random.uniform(- outer_mostR, outer_mostR)
#        y = random.uniform(- outer_mostR, outer_mostR)
#        z = random.uniform(- outer_mostR, outer_mostR)
#        
#        #Calculate which values to keep based on the equation of a circle
#        if ((math.sqrt(x**2+y**2) - R)**2 + z**2 > r**2):
#            out_vol.append([x,y,z])
#        else:
#            vol.append([x,y,z])
#      
#    #Returns the value of a torus with majr radius 1 and minor radius 0.5 (default arg)
#    # Part a
#    # The exact volume of a torus with major radius 1 and minor radius 0.5  is 4.934802201
#    # len(area)/N * (2*(outer_mostR))**3 where the fraction represend point inside the torus
#    # wrt to the cubic region. The product comes from the volume of the cube.
#    print("Volume of a torus with major radius {0} and minor radius {1} is {2}".format(R,r, len(vol)/N * (2*(outer_mostR))**3))
#    
#    #Conversion from list to arrays for easy plotting
#    vol = np.transpose(np.array(vol))
#    out_vol = np.transpose(np.array(out_vol))
#    
#    #Calculate boundary of the circle  
#    theta = np.linspace(0, 2.*np.pi, 100)
#    phi = np.linspace(0, 2.*np.pi, 100)
#    theta, phi = np.meshgrid(theta, phi)
#    x = (R + r*np.cos(theta)) * np.cos(phi)
#    y = (R + r*np.cos(theta)) * np.sin(phi)
#    z = r * np.sin(theta)
#    
#    #Part B
#    fig = plt.figure(figsize=[12,12])
#    ax1 = fig.add_subplot(111, projection='3d')
#    ax1.scatter(out_vol[0],out_vol[1],out_vol[2],'.',color='red', zorder = 1)
#    ax1.plot_wireframe(x, y, z, rstride=5, cstride=5, color='k',  zorder = 2)
#    ax1.scatter(vol[0],vol[1],vol[2],'.',color='blue',  zorder = 3)
#    ax1.set_zlim(- outer_mostR, outer_mostR)
#    ax1.set_xlim(- outer_mostR, outer_mostR)
#    ax1.set_ylim(- outer_mostR, outer_mostR)
#    ax1.text2D(0.325, 0.95, "Volume of a torus using Monte Carlo integration", transform=ax1.transAxes)
#    ax1.set_xlabel('X axis')
#    ax1.set_ylabel('Y axis')
#    ax1.set_zlabel('Z axis')
#    plt.show()
#    
#    if R > r:
#        print('This is a regular torus (ring torus)')
#    if R == r:
#        print('This is a horn torus (no hole)')
#    if (R < r) & (R != 0):
#        print('This is a spindle torus (self intersecting)')
#    if R == 0:
#        print('This is a degenerate form of torus which is sphere with radius {0}'.format(r))
#    
#
#volume_torus()