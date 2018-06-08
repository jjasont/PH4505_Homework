# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:36:49 2018
Updated on Tue Mar 13 10:20    2018
- Change of normalization calculation, multiply by dx
- Lengthen the time from for running the ODE for 2b to remove transient state
- Automate the process to determine the state for shooting method question 3d
#2 Update on Tue Mar 13 11:30  2018
- Modulo of 2*pi introduced for theta which omega is zero (within tolerance), question 2d.
- Change of tolerance level for zero omega in question 2d.
- Change default parameter for nmin and nmax state of question 3b and 3c from 1, 3 to 0, 2 respectively

Question 2 (9/10)
For part c, you only need one single plot to show the sensitivity to initial conditions. 
You are missing labels for part d.

Question 3 (9/10)
In line 262, you did not define or call dx in your function.

@author: Jason Tanuwijaya
"""

#%%










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

# Define the function to be solve for pendulum
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
##############################################################
#%%
print('Question 2 Part b running')
choiceFd = np.array([40,60,65,70,80]) # The proposed magnitude of driving force
fig = plt.figure(figsize = (7,10))
gamma, pendC = 0.1, 10
tmin = 0
tmax = 1000
tobs = 20
tn = 20000
t = np.linspace(tmin, tmax, tn)
for i in range(0,len(choiceFd)):
    sol = odeint(pend, y0, t, args=(gamma, pendC, choiceFd[i]))
    plt.subplot(510+i+1) #subplot for each driving force value. 511 for the first, 512 for the 2nd...
    startPlotIndex = np.where(abs(t - (tmax - tobs)) < 0.05) # Find the entry 20 second from the last time
    startPlotIndex = np.asscalar(startPlotIndex[0][0])
    plt.plot(t[startPlotIndex:len(t)], sol[startPlotIndex:len(t), 1], 'g', label='omega(t)')
    #plt.plot(t,sol[:,1],'b',label='fullOmega(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.title('Fd = %i' %choiceFd[i])
    plt.grid()
plt.tight_layout()
plt.show()
print('Question 2 Part b completed')
##############################################################
#%%
print('Question 2 Part c running')
fig1 = plt.figure(figsize = (7,10))
nPoint = [10, 50, 100, 250] # Containt different resolution point for value between 50 to 55
minFd, maxFd = 50, 55 # Preferred value of driving force range
deltaTheta = 1e-8
gamma, pendC = 0.1, 10.0 # Constant Initialization
tmin = 0.0
t = np.linspace(tmin, tmin+50.0, tn) # Timeframe of observation is 50

y0 = [1.0, 0.0] #ODE Initial condition
y0prime = y0[:]
y0prime[0] += deltaTheta #y0prime is the perturbated 

for nPoint, i in zip(nPoint,range(0,len(nPoint))): # Loop for different resolution to compare
    plt.subplot(410+i+1)
    pointFd = np.linspace(minFd, maxFd, nPoint)
    
    deltaOmega = np.zeros((len(pointFd),1), dtype = float)
    for Fd, index in zip(pointFd, range(0,len(pointFd))):
        sol0 = odeint(pend, y0, t, args=(gamma, pendC, Fd)) # The original ODE solution
        sol1 = odeint(pend, y0prime, t, args=(gamma, pendC, Fd)) # The perturbed ODE solution
        diffTheta = abs(sol0[len(t)-1,1] - sol1[len(t)-1,1])
        deltaOmega[index] = diffTheta
    plt.plot(pointFd, deltaOmega, 'k.', label='Delta omega(T), T = 50')
    plt.xlabel('Fd')
    plt.title('Number of Point = %i' % nPoint)

plt.tight_layout()
plt.show()
# Fluctuation of the angular velocity difference observed as the magnitude of driving force get above 52.7
# Different resolution/number of point for delta shows that perturbation can be better observed with higher
# number of point observed betwen 50 to 55

print('Question 2 Part c completed')
##############################################################
#%%
print('Question 2 Part d running')
minFd, maxFd, nFd = 50, 80, 100
gamma, pendC = 0.1, 10.0
pointFd = np.linspace(minFd, maxFd, nFd)
tmin = 0.0
t = np.linspace(tmin, tmin+1000.0, 1e6) # Run up to 1e6 to remove the effects of transient state and gaining more data point
tol = 1e-4 # The tolerance level to say the omega is zero
y0 = [1, 0.0] # ODE initial condition again
zerosTheta = [] # Empty array to contain the theta which omega is zero
for Fd, index in zip(pointFd, range(0,len(pointFd))):
    #print('This will take  a while. Working for driving Force {0:.2f}'.format(Fd))
    sol2 = odeint(pend, y0, t, args=(gamma, pendC, Fd))
    if sol2[0,1] < 1e-6: # Float limitation. == 0.0 is no recommended). Check whether initial condition of omega is 0 within 1e-6
        sol2_noIn = sol2[1:,] # Remove initial condition when initial omega is 0 (within 1e-6)
    else:
        sol2_noIn = sol2[:] # Proceed as normal
    # Find the entries which omega is less than the tolerance value of 0 (used is tol = 1e-3)
    zero_valTheta = sol2_noIn[abs(sol2_noIn[:,1]) < tol] 
    zerosTheta.append(np.remainder(zero_valTheta, 2*np.pi))
    plt.plot(np.full((len(zero_valTheta),1), Fd), zerosTheta[index][:,0], 'b.')
plt.show()
print('Question 2 Part d completed')
## ================================================================================================ ##
#%% Question 3
#%% Question 3 part a
#from scipy import *
print('Question 3 Part a running')
import numpy as np
import matplotlib.pyplot as plt

# Define the energy of QHO
def energyQHO(n, hbar = 1, omega = 1):
    E = (n + 0.5)*hbar*omega # hbar and omega equal 1
    return E

# Define the potential of QHO
def potQHO(x, hbar = 1, omega = 1, m = 1):
    x = np.array(x)
    pot = m*omega**2 * x**2/2 # hbar, omega and m equal 1
    return pot


def numerov1(nstate = 1, xmin = -6, xmax = 6, N = 60000, energyTrial = 1, energyKnown = True): 
    dx = (xmax-xmin)/N
    dx2 = dx*dx
    psi = np.zeros(N)
    global V
    V = np.zeros(N)
    psi[0] = 0.0
    if nstate%2 == 0:    
        psi[1] = 0.01 # When the nstate is even, start with positive initial value
    else:
        psi[1] = -0.01 # When the nstate is odd, start with negative initial value
    
    if energyKnown: # Assuming the energy is known
        E = energyQHO(nstate)
        for i in range(N):
            x = xmin+i*dx
            V[i] = potQHO(x)
       
        for i in range(1,N-1):
            f_nought = 2.0 * (1.0 + ( 5.0 *dx2 / 12.0 )*2.0*(V[i] - E)) #fk(V[i],E)
            f_plus = (1.0 - (dx2 / 12.0) * 2*(V[i+1] - E))
            f_min = (1.0 - (dx2 / 12.0) * 2*(V[i-1] - E))
            psi[i+1] = (f_nought*psi[i] - f_min*psi[i-1])/f_plus
# =============================================================================
#         norm=0.0
#         for i in range(N):
#             norm += psi[i]*psi[i]
#         psi[:]=psi[:]/np.sqrt(norm)
#         return psi
# =============================================================================
    else: # For shooting method
        E = energyTrial
        for i in range(N):
            x = xmin+i*dx
            V[i] = potQHO(x)
        for i in range(1,N-1):
            f_nought = 2.0 * (1.0 + ( 5.0 *dx2 / 12.0 )*2.0*(V[i] - E)) #fk(V[i],E)
            f_plus = (1.0 - (dx2 / 12.0) * 2*(V[i+1] - E))
            f_min = (1.0 - (dx2 / 12.0) * 2*(V[i-1] - E))
            psi[i+1] = (f_nought*psi[i] - f_min*psi[i-1])/f_plus
    norm=0.0
    for i in range(N):
        norm += psi[i]*psi[i]*dx
    psi[:]=psi[:]/np.sqrt(norm)
    return psi

print('Question 3 Part a completed')
#%%
print('Question 3 Part b running')
xmin = -6
xmax = 6
N = 60000
x = np.linspace(xmin,xmax,N) 

def plotStateNumerov(nmin = 0, nmax = 2):
    fig2 = plt.figure(figsize = (14,10))
    psi_ = []
    for n in range(nmin, nmax+1):
        psi = numerov1(n)
        psi_.append(psi)
        #print(np.abs(np.sum(psi))**2)
        #plt.subplot(121)
        plt.title('Schrodinger Equation of QHO System with Numerov')    
        E = energyQHO(n)
        #print(E)
        plt.plot(x,psi+E, label = 'QHO Wave Function n = %d' %n)
        #plt.xlim(xmin,xmax)
        #plt.ylim(min(psi)-0.2 , max(psi)+ 0.2)
        plt.ylabel("$\psi(x)$")
        plt.xlabel("x")
        plt.grid()
        plt.legend(loc = 'best')
        # Plot Energy Level
#        plt.subplot(122)
#        plt.plot(x, np.full((N),E), label = 'Eigenenergy for n = {0}, E = {1}'.format(n, E))
#        plt.xlabel("x")
#        plt.legend(loc = 'best')
    #plt.subplot(121)
    plt.plot(x, V, 'k-', label = 'Harmonic Oscillator Potential')
    plt.ylabel('Energy')
    plt.title('Harmonic Oscillator Potential + Energy')
    plt.legend(loc = 'best')
    plt.grid()
    plt.show()
    return psi_
    
plotStateNumerov(nmax = 12)
print('Question 3 Part b completed')
#%%
print('Question 3 Part c running')
def plotStateScipy(nmin = 0, nmax = 2):
    fig3 = plt.figure(figsize = (14,5))
    from scipy.integrate import odeint
    for n in range(nmin, nmax+1):
        plt.subplot(121)
        plt.title('Schrodinger Equation of QHO System with Scipy odeint')
        Energy = energyQHO(n)
        if n%2 == 0:
            y0 = [1.0, 0.0]
        else:
            y0 = [-1.0, 0.0] # Odd  state start from negative initial value / left
        QHOsol = odeint(schrodinger, y0, x, args=(Energy, potQHO))
        psi = QHOsol[:,0]
        norm = 0.0
        for i in range(N):
            norm += psi[i]*psi[i]*dx
        psi[:]=psi[:]/np.sqrt(norm)
        plt.plot(x,psi, label = 'QHO Wave Function n = %d' %n)
        plt.xlim(xmin,xmax)
        plt.ylabel("$\psi(x)$")
        plt.xlabel("x")
        plt.grid()
        plt.legend(loc = 'best')
        # Plot Energy Level
        plt.subplot(122)
        plt.plot(x, np.full((N),Energy), label = 'Eigenenergy for n = {0}, E = {1}'.format(n, Energy))
        plt.xlabel("x")
        plt.legend(loc = 'best')
    plt.subplot(122)
    plt.plot(x, potQHO(x), 'k-', label = 'Harmonic Oscillator Potential')
    plt.ylabel('Energy')
    plt.title('Harmonic Oscillator Potential + Energy')
    plt.legend(loc = 'best')
    plt.grid()
    plt.show()
        
def schrodinger(y, x, Energy, Potential):
    psi, phi = y
    dydx = [phi, 2*(Potential(x) - Energy)*psi]
    return dydx

plotStateScipy()
print('Question 3 Part c completed')
#%%
print('Question 3 Part d completed')
def shootingMethodNumerov(energyTrial, N = 10000):
    dE = 0.01
    tol = 1e-6
    E1 = energyTrial
    xmin = -6.0
    xmax = 6.0
    N = 10000
    # Initial guess
    psi1 = numerov1(xmin = xmin, xmax = xmax, N = N, energyTrial = E1, energyKnown = False)
    b1 = psi1[N-1]
    while True:
        E2 = E1 + dE
        psi2 = numerov1(xmin = xmin, xmax = xmax, N = N, energyTrial = E2, energyKnown = False)
        b2 = psi2[N-1]
        if b1*b2 < 0:
            print('The energy lays between {0:.4f} and {1:.4f}'.format(E1,E2))
            break    #correct sign change    
        else:
            E1=E2
            b1=b2
            
    # Bisection method
    while True:
        E0 = (E1+E2)/2 # Middle point
        psi1 = numerov1(xmin = xmin, xmax = xmax, N = N, energyTrial = E0, energyKnown = False)
        b0 = psi1[N-1]
        if (abs(b0) < tol): # The end term of the wave function is less than tol value to be zero
            break
        else:
            if (b0*b1 < 0):
                E2=E0 # Set middle point/bisection point as UPPER bound
                b2=b0
            else:
                E1=E0 # Set middle point/bisection point as LOWER bound
                b1=b0
    from math import floor
    print('This is state n = {0:d}. The eigenenergy is {1:.12f}. This differ by {2:.4E}'.format(floor(E0),E0,abs(E0 - floor(E0)-0.5)))
    if floor(E0)%2 == 0:
        plt.plot(np.linspace(xmin, xmax, N), -psi1) #Flipped the sign of the wave function if even state
    else:
        plt.plot(np.linspace(xmin, xmax, N), psi1)
    plt.show()
    return E0

# For state n = 1
E0 = shootingMethodNumerov(energyTrial = 1)
# =============================================================================
# This return the energy for the state n = 1
# Hence, the difference are (1+0.5) - E0
# =============================================================================
#print('The difference for trial energy = 1 (result in energy of around 1.5, state n = 1) with eigenenergy at state n = 1 is {0:.4E}'.format(abs(1.5-E0)))

# For state n = 2
E0 = shootingMethodNumerov(energyTrial = 2)
# =============================================================================
# This return the energy for the state n = 2
# Hence, the difference are (2+0.5) - E0
# =============================================================================
#print('The difference for trial energy = 2 (result in energy of around 2.5, state n = 2) with eigenenergy at state n = 1 is {0:.4E}'.format(abs(2.5-E0)))
print('Question 3 Part d completed')