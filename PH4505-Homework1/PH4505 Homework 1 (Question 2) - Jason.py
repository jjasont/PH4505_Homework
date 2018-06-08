# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:22:39 2018
Question 2 (9/10)

Your first force value is nan. This is a result of the way we are doing the integration for the question. 
As you increase n, you should also see that the force goes to zero as we approach z=0. 
This is unphysical. You need to take these into consideration.
@author: Jason (U1440158A)
"""
#%%
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