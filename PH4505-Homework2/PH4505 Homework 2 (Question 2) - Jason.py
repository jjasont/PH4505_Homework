# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:36:49 2018

@author: Jason Tanuwijaya
"""

#%%
# Question 2
#%%
print('Question 2 Part a running')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
tmin, tmax, tn = 0,30,1001 # Define the value of starting time, finishing time and the resolution/[point]
gamma = 0.25 # Choose arbitrary value of damping constant
pendC = 2 # Choose arbitrary value of pendulum constant/omega_0
y0 = [1, 0.0] # nearly vertical, omega at 0 is 0, [theta, omega]

t = np.linspace(tmin, tmax, tn) #t will be the time of observation

def pend(y, t, gamma, pendC, Fd = 1):
    theta, omega = y
    dydt = [omega, Fd*np.cos(2*np.pi*t) - 2*gamma*omega - pendC**2 *np.sin(theta)]
    return dydt


sol = odeint(pend, y0, t, args=(gamma, pendC, 1))


plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
print('Question 2 Part a completed')
#%%
print('Question 2 Part b running')
choiceFd = np.array([40,60,65,70,80]) # The proposed magnitude of driving force
fig = plt.figure(figsize = (7,10))
gamma, pendC = 0.1, 10
tmin = 0
t = np.linspace(tmin, tmin+20, tn)
for i in range(0,len(choiceFd)):
    sol = odeint(pend, y0, t, args=(gamma, pendC, choiceFd[i]))
    plt.subplot(510+i+1) #subplot for each driving force value. 511 for the first, 512 for the 2nd...
    #plt.plot(t, sol[:, 0], 'b', label='theta(t)')
    plt.plot(t, sol[:, 1], 'g', label='omega(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    #plt.ylabel('omega(t)')
    plt.title('Fd = %i' %choiceFd[i])
    plt.grid()
plt.tight_layout()
plt.show()
print('Question 2 Part b completed')

#%%
print('Question 2 Part c running')
fig1 = plt.figure(figsize = (7,10))
nPoint = [10, 50, 100, 250] # Contain differen resolution point for value between 50 to 55
for nPoint, i in zip(nPoint,range(0,len(nPoint))):
    plt.subplot(410+i+1)
    minFd, maxFd, nFd = 50, 55, nPoint
    pointFd = np.linspace(minFd, maxFd, nFd)
    deltaTheta = 1e-8
    gamma, pendC = 0.1, 10.0
    
    tmin = 0.0
    t = np.linspace(tmin, tmin+50.0, tn)
    plt.subplot(510+i+1)
    #y0 = [np.pi - 0.1, 0.0]
    y0 = [1.0, 0.0]
    y0prime = y0[:]
    y0prime[0] += deltaTheta
    deltaOmega = np.zeros((len(pointFd),1), dtype = float)
    for Fd, index in zip(pointFd, range(0,len(pointFd))):
        sol0 = odeint(pend, y0, t, args=(gamma, pendC, Fd))
        sol1 = odeint(pend, y0prime, t, args=(gamma, pendC, Fd))
        #print('The final value of omega is {0:f}'.format())
        #print(index)
        #print('The final value of omega deviated is {0:.3f}'.format(sol1[len(pointFd)-1,1]))
        diffTheta = abs(sol0[len(pointFd)-1,1] - sol1[len(pointFd)-1,1])
        deltaOmega[index] = diffTheta
    plt.plot(pointFd, deltaOmega, 'k.', label='Delta omega(T), T = 50')
    plt.xlabel('Fd')
    plt.title('Number of Point = %i' % nPoint)

plt.tight_layout()
plt.show()
# Disturbance of the angular velocity observed as the magnitude of driving force get above 52
# Different resolution/number of point for delta shows that perturbation can be observed clearly with higher
# number of point observed

print('Question 2 Part c completed')
#%%
print('Question 2 Part d running')
minFd, maxFd, nFd = 50, 80, 100
gamma, pendC = 0.1, 10.0
pointFd = np.linspace(minFd, maxFd, nFd)
tmin = 0.0
t = np.linspace(tmin, tmin+100.0, 1e6) # Run up to 1e6 to remove the effects of transient state
tol = 1e-3 # The tolerance level to say the omega is zero
y0 = [1, 0.0]
zerosTheta = []
for Fd, index in zip(pointFd, range(0,len(pointFd))):
    sol2 = odeint(pend, y0, t, args=(gamma, pendC, Fd))
    if sol2[0,1] < 1e-6: # Float limitation. == 0.0 is no recommended)
        sol2_noIn = sol2[1:,] # Remove initial condition when initial omega is 0 (within 1e-6)
    else:
        sol2_noIn = sol2[:]
    # Find the entries which omega is less than the tolerance value of 0 (used is 1e-3)
    zero_valTheta = sol2_noIn[abs(sol2_noIn[:,1]) < tol] 
    zerosTheta.append(zero_valTheta)
    plt.plot(np.full((len(zero_valTheta),1), Fd), zerosTheta[index][:,0], 'b.')
plt.show()
print('Question 2 Part d completed')