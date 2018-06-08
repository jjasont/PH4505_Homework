# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:45:32 2018

Question 1 (7/10)

For part b, You need to plot all of the eigenvalues as we want the energy-level diagram. 
Also, do not take the absolute value of the energy. 
For part d, your values are a little weird as IPR should be less than 1. 
Also, your graph should be smooth. For part e, you need to use markers to indicate the states you are plotting. 
The bands are indicated by the energies clumping together. 
You are right that the rationality of the alpha affects the out of band state. 
Out of band states only occur when alpha is irrational.


Question 3 (9.5/10)

Overall very good. Also, ODEs can be numerically unstable. 
However, sometimes using the matrix method might consume too much memory. 
For those cases, solving the ODE is still better.

@author: Jason Tanuwijaya
"""

#%% Question 1
#% Question 1a
def harperHamiltonian(N = 3, W = 1, psi = 0, alpha = 1):
    # Import necessary function
    from numpy import arange, ones
    from scipy.sparse import dia_matrix
    
    def potential(vector_n, W, psi, alpha):
        # Import necessary function to calculate the potential vector
        from numpy import cos, pi
        return W*cos(2*pi*alpha*vector_n + psi)
    
    one_vec = ones(N) # Component -1 and +1 off diagonal
    potential_vec = potential(arange(0,N), W, psi, alpha) # Main diagonal component
    
    # Final matrix 
    hamiltonian_matrix = dia_matrix(([one_vec,potential_vec,one_vec], [-1,0,1]),shape=(N,N))

    return hamiltonian_matrix
#%%  
#% Question 1b
import numpy as np
import matplotlib.pyplot as plt 

def functionQ1b(numev = 3):
    from scipy.sparse.linalg import eigs
#    Import package if not declared globally
#    import numpy as np
#    import matplotlib.pyplot as plt 
    N = 199
    W = 2.0
    psi = 0.0
    alpha = np.linspace(0.0,2.0,300)
    E = []
    
    for a in alpha:
        hamiltonian_matrix = harperHamiltonian(N,W,psi,alpha = a)
        energy, wavefunc = eigs(hamiltonian_matrix, k = numev, sigma = -1.0)
        E.append(energy)
        
    plt.figure(figsize = (10,7.5))
    E = np.array(E)
    
    legend_string = []
    for i in range(1,numev+1):
        legend_ = 'n = ' + str(i)
        legend_string.append(legend_)
    
    plt.plot(alpha, E, '.')
    #plt.legend(legend_string)
    plt.xlabel('alpha')
    plt.ylabel('Energy')
    plt.title('Energy vs Alpha - Harper Model')
    return E

print('Question 1 Part b running')
E = functionQ1b(numev = 197)
print('Question 1 Part b finished')
#%%
#% Question 1c
def functionQ1c(numev = 3):
#    Import package if not declared globally
#    import numpy as np
#    import matplotlib.pyplot as plt 
    from scipy.sparse.linalg import eigs
    N = 199
    alpha = 1.618034 
    psi = 0.0
    W = np.linspace(1.2,2.5,15)
    wave_f = []
    plt.figure(figsize = (30,30))
    counter = 1
    for w in W:
        # 5 by 3 Subplot
        plt.subplot(5,3,counter)
        counter += 1
        hamiltonian_matrix = harperHamiltonian(N,W = w,psi = psi,alpha = alpha)
        energy, wavefunc = eigs(hamiltonian_matrix, k = numev, sigma = -1.0)
        wave_f.append(wavefunc)
        label_fig = "Ground State, W = " + str(w)
        plt.plot(np.arange(0,N), abs(wavefunc[:,0])**2, label = label_fig)
        plt.xlabel('n')
        plt.ylabel('|psi(n)|^2')
        plt.title('Probability Density')
        plt.legend()
    plt.tight_layout()

print('Question 1 Part c running')
functionQ1c()
print('Question 1 Part c finished')

# For small W, the value of the wave function increase along with the position
# With increasing W, the fluctuation become more gentle and eventually the wave function become
# symmetric for value of W around 2 and inhibiting a peak.
# The point where it starting to be more localized at W between 1.85 and 1.95
# The non peak value getting smoother when W increased until W = 2.2 - 2.3, it doesn't change much
# The peak observed to be at the position n = 116

#%% Question 1d
import numpy as np
def IPR(wavefunc):
	#import numpy as np
	return np.mean(np.sum(abs(wavefunc)**4))

def functionQ1d(numev = 3, numW = 100):
    #import numpy as np
    import matplotlib.pyplot as plt 
    from scipy.sparse.linalg import eigs
    N = 199
    alpha = 1.618034 
    psi = 0.0
    W = np.linspace(1.2,2.5,numW)
    IPR_allW = []
    plt.figure(figsize = (10,10))
    for w in W:
        hamiltonian_matrix = harperHamiltonian(N,W = w,psi = psi,alpha = alpha)
        energy, wavefunc = eigs(hamiltonian_matrix, k = numev, sigma = -1.0)
        #wave_f.append(wavefunc)
        #label_fig = "Ground State, W = " + str(w)
        IPR_W = IPR(wavefunc)
        IPR_allW.append(IPR_W)
        #plt.plot(np.arange(0,N), abs(wavefunc[:,0])**2, label = label_fig)
    plt.plot(W, IPR_allW)
    plt.xlabel('W')
    plt.ylabel('IPR')
    plt.title('IPR vs W')
    return [W, IPR_allW]
    #plt.legend()
    #plt.tight_layout()

print('Question 1 Part d running')
[W, IPR_result] = functionQ1d(numev = 8, numW = 250)
y = np.diff(IPR_result) # Shortcut with differentiation

#plt.plot(W[1:],y)
print('Critical W is around {0:.4f}'.format(W[np.argmin(y)+1])) # Critical W 
print('Dip in IPR is around {0:.4f} at critical W'.format(IPR_result[np.argmin(y)+1])) # Critical W 
print('Question 1 Part d finished')
#print(IPR_result[np.argmin(y)+1]


# IPR is large for more localized states and small for extended states
# Critical W suspected to be in the range of 1.85 to 1.95
# From simple check with W[np.argmin(y)+1] , it returned value 0f 1.88
# using simple difference equation value
#%% Question 1e

def functionQ1e(numev = 1, alpha = 1, threshold = 'quantile', q = 5, phase_plot = 100):
    # --------------- Setup ------------------
    from scipy.sparse.linalg import eigs
    import numpy as np
    import matplotlib.pyplot as plt 
    N = 199
    W = 1.0
    psi = np.linspace(-np.pi,np.pi,201)
    #alpha = np.pi #np.sqrt(np.pi) # np.sqrt(2) 
    E = []
    wavef_ = []
    for phase in psi:
        #print(a)
        #print(phase)
        hamiltonian_matrix = harperHamiltonian(N,W,psi = phase,alpha = alpha)
        energy, wavefunc = eigs(hamiltonian_matrix, k = numev, sigma = -1.0)
        E.append(energy)
        wavef_.append(wavefunc)
    E = np.array(E)
    wavef_ = np.array(wavef_)
    # ------------------------------------------------------------------

    # ------------- Subplot for every 5 plot to 2 by 2 subplot -------------
#    legend_string = []
#    counter = 0
#    plt.figure(figsize = (10,10))
#    for data in range(0, numev):
#        legend_ = 'n = ' + str(data)
#        legend_string.append(legend_)
#        if ((data+1) % 5) == 0:
#            counter += 1
#            plt.subplot(2,2,counter)
#            plt.plot(psi, abs(E[:,np.arange(data-4,data+1)]), '--')#np.mean(E, axis = 1))
#            plt.legend(legend_string)
#            legend_string = []
#     
#    plt.show()
    # -----------------------------------------------------------------
    
    # ------------- Use correlation to detect band ----------------------
    gamma = np.corrcoef(np.transpose(E))
    gamma = gamma.real
    plt.imshow(gamma)
    plt.title('Correlation Matrix Plot')
    plt.show()
    sum_gamma = np.sum(gamma, axis = 0)
    # Large sum_gamma means some similarity with other energy band
    # Low sum_gamma mean disimilar with other energy band
    if threshold == 'quantile':
        threshold_gamma = np.percentile(sum_gamma, q) # np.mean(sum_gamma) - np.std(sum_gamma)
    elif threshold == 'mean':
        threshold_gamma = np.mean(sum_gamma) - np.std(sum_gamma)
    outofband_inx = np.argwhere(sum_gamma < threshold_gamma).flatten()
    # -----------------------------------------------------------------
    
    
    # ------------ Plot out-of-band wave function ---------------------------
    legend_string = []
    for item in outofband_inx:
        legend_ = 'n = ' + str(int(item))
        legend_string.append(legend_)
    plt.figure(figsize = [10,10])
#    plt.subplot(2,1,1) # Uncomment with section (*)
    plt.plot(np.transpose(abs(wavef_[phase_plot,:,outofband_inx])**2))
    plt.xlabel('x')
    plt.ylabel('|psi(x)|^2')
    plt.title(" 'Out-of-band' state ")
    plt.legend(legend_string)
    
    
    # ------------ Plot in-of-band wave function ---------------------------
    band_inx = np.setdiff1d(np.arange(0,numev), outofband_inx) # Which n in band form
    
    # Continue appending from the previous legend out-of-band
    for i in band_inx:
        legend_ = 'n = ' + str(i)
        legend_string.append(legend_)
        
    # Section (*)
#    plt.subplot(2,1,2)
#    plt.plot(np.transpose(abs(wavefunction[5,:,band_inx])**2))
#    plt.xlabel('x')
#    plt.ylabel('|psi(x)|^2')
#    plt.title(" 'In-band' state ")
#    plt.legend(legend_string)
#    
#    plt.show()
    
    # ----------------- Plot out-of-band energy band first -----------
    plt.figure(figsize = (10,10))
    plt.plot(psi, E[:,outofband_inx], '+')
    # -----------------------------------------------------------------
    
    # --------------- Plot the rest energy value ------------------
    plt.plot(psi, E[:,band_inx], '.')#np.mean(E, axis = 1))
    plt.xlabel('psi')
    plt.ylabel('Energy')
    plt.legend(legend_string,loc='center right', bbox_to_anchor=(1.15, 0.5))
    plt.title('Energy vs Psi - Harper Model')
    plt.ylim([np.min(E) - 0.1, np.max(E)+0.1])
    plt.show()
    # ------------------------------------------------------------------
    
    return [psi, E, wavef_, gamma]
#%%
print('Question 1 Part e running')
alpha = np.pi/10
[psi, E, wavefunction, gamma] = functionQ1e(numev = 20, alpha = alpha, q = 5, phase_plot = 170)
print('Question 1 Part e finished')
#%%
# Honestly, I have no idea what does it mean by cluster into "bands"
# I assume that it's the behaviour that it form a similar eigenenergy pattern/high correlation
# For 20 eigenenergy and alpha = pi/10, it observed to inhibit 2 main band laying from energy ~0.42 to 0.52
# and energy ~0.65 to ~0.7
# The most extreme energy pattern observed for the 0-th state, 10-th state and 11-th state
# This identification may change depending on the algorithm of the eigs function determining the lowest eigenvalue
# Correlation matrix plotted to see better pattern and indeed the 0-th, 10-th and 11-th state result in the most extreme difference
# For the correlation matrix plot, look for the row with the most blue. It's the most out of band
# However, the behaviour changed as the value of alpha (which decided to be irrational) changed.
# Energy plot with '+' is for the out-of-band
# The most intriguing part is that for n = 0, the energy tend to deviate the most.
# It's hard to automate the pindown the out-of-band state

#%% Question 3
#% Question 3a
def qhoHamiltonian(N = 100, L = 5, m = 1, hbar_ = 1, omega = 1):
    # Import necessary function
    from numpy import ones, linspace
    from scipy.sparse import dia_matrix
    
    dx = L/(N+1.0)
    x  = linspace(dx, L-dx, N)-L/2
    
    def potential(vector_n, m, hbar_, omega):
        return (m*omega*vector_n**2)/2
    
    one_vec = ones(N) # Component -1 and +1 off diagonal
    #two_vec = -2*ones(N)
    potential_vec = potential(x, m, hbar_, omega) # Main diagonal component
    
    # Final matrix 
    #hamiltonian_matrix = dia_matrix(([one_vec,-0.5*two_vec/(dx*dx) + potential_vec,one_vec], [-1,0,1]),shape=(N,N))
    
    hamiltonian_matrix = dia_matrix(([one_vec, -2*one_vec, one_vec], [-1,0,1]), shape=(N,N))
    hamiltonian_matrix *= -0.5/(dx*dx)
    hamiltonian_matrix += dia_matrix(([potential_vec], [0]),shape=(N,N))
    #hamiltonian_matrix *= -0.5/(dx*dx)
    
    return [hamiltonian_matrix, x, dx]

#%%
print('Question 3 Part b running')
def functionQ3b(numev = 3, N = 25000, verbose = True):#, scale = 1.002455378):
    from scipy.sparse.linalg import eigsh
    import numpy as np
    import matplotlib.pyplot as plt 
    L = 12 #17555 * N/ 25000 * scale
    [hamiltonian_matrix, x, dx] = qhoHamiltonian(N = N, L = L)
    # Tradeoff between N and L
    # Small L result in more detailed plot, yet large/magnified energy
    # Large/proportionate L wrt to N result in rought plot, yet rather accurate eigenenergy
    energy, wavefunc = eigsh(hamiltonian_matrix, k = numev, sigma = -1.0)
    #energy = energy/constantEig
    
    if verbose:
        if numev == 3:
            print("The calculated solutions are:")
            print(energy[0], energy[1], energy[2])
        elif numev == 1:
            print("The calculated solutions are:")
            print(energy[0])
        else:
            print(energy)
        plt.figure(figsize = (10,7.5))
    
    if verbose:
        for i in range(0,numev):
            title = 'wavefunc vs x, n = ' + str(i) + ', E = ' + str(energy[i])
            #legend_string.append(legend_)
            plt.subplot(numev,1,i+1)
            plt.plot(x, wavefunc[:,i], '.-')
            #plt.legend(legend_string)
            plt.xlabel('x')
            plt.ylabel('wavefunc')
            plt.title(title)
            #plt.xlim((-2, 2))
        
        #'n = '
        
        plt.tight_layout()
        if numev == 3:
            print("The analytical solutions are:")
            print(1/2, 3/2, 5/2)
        elif numev == 1:
            print("The analytical solutions are:")
            print(1/2)
    return [energy, wavefunc.real, x]
    
[E,wavefunc, x] = functionQ3b(numev = 3)
print('Question 3 Part b finished')
#%%
print('Question 3 Part c running')
import numpy as np
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


def shootingMethodNumerov(energyTrial, N = 10000, tol = 1e-6, verbose = True):
    import matplotlib.pyplot as plt 
    dE = 0.01
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
            if verbose:
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
    
    if verbose:
        print('This is state n = {0:d}. The eigenenergy is {1:.12f}. This differ by {2:.4E}'.format(floor(E0),E0,abs(E0 - floor(E0)-0.5)))
        if floor(E0)%2 == 0:
            plt.plot(np.linspace(xmin, xmax, N), -psi1) #Flipped the sign of the wave function if even state
        else:
            plt.plot(np.linspace(xmin, xmax, N), psi1)
        plt.show()
    return E0

# For state n = 1
E0 = shootingMethodNumerov(energyTrial = 4.89)
print('Question 3 Part c finished')
#%% Question 3d
#import timeit
def functionQ3d(k = 3, N = 10000, tol = 1e-3):
    E_Numerov = []
    for i in range(0,k):
        E_Numerov.append(shootingMethodNumerov(energyTrial = i, N = N, tol = tol,verbose = False))
    E_Numerov = np.array(E_Numerov)
    print('Energy found via Numerov are')
    print(E_Numerov)
    print('Deviation from analytical result are ')
    print(abs(E_analytic - E_Numerov)/E_analytic)

def functionQ3d1(k = 3, N = 10000):
    [E_Matrix, _, _] = functionQ3b(numev = k, N = N, verbose = False)
    print('Energy found via Matrix are')
    print(E_Matrix)
    print('Deviation from analytical result are ')
    print(abs(E_analytic - E_Matrix)/E_analytic)

if __name__ == '__main__':
    #E_analytic = np.array([0.5, 1.5, 2.5, 3.5])
    k = 1
    E_analytic = np.arange(0.5, k + 0.5, 1)
    import timeit
    print("================== BENCHMARK 1 - Ground State, ================")
    print('Numerov Method - Shooting. Time Elapse = {0}'.format(timeit.timeit("functionQ3d(k = 1, N = 10000)", setup="from __main__ import functionQ3d", number = 1)))
    print('Matrix Method. Time Elapse = {0}'.format(timeit.timeit("functionQ3d1(k = 1, N = 10000)", setup="from __main__ import functionQ3d1", number = 1)))
    
    k = 4
    E_analytic = np.arange(0.5, k + 0.5, 1)
    print("================== BENCHMARK 2 - 4 First State ================")
    print('Numerov Method - Shooting for 3 ground state. Time Elapse = {0}'.format(timeit.timeit("functionQ3d(k = " + str(k) + ")", setup="from __main__ import functionQ3d", number = 1)))
    print('Matrix Method for 3 ground state. Time Elapse = {0}'.format(timeit.timeit("functionQ3d1(k = " + str(k) + ")", setup="from __main__ import functionQ3d1", number = 1)))
    
    k = 10
    E_analytic = np.arange(0.5, k + 0.5, 1)
    print("================== BENCHMARK 3 - 10 First State ================")
    print('Numerov Method - Shooting for 3 ground state. Time Elapse = {0}'.format(timeit.timeit("functionQ3d(k = " + str(k) + ")", setup="from __main__ import functionQ3d", number = 1)))
    print('Matrix Method for 3 ground state. Time Elapse = {0}'.format(timeit.timeit("functionQ3d1(k = " + str(k) + ")", setup="from __main__ import functionQ3d1", number = 1)))
    
    k = 3
    E_analytic = np.arange(0.5, k + 0.5, 1)
    print("================== BENCHMARK 4 - Ground State - 20k pts vs 320k pts ================")
    print('Numerov Method - Shooting for 3 ground state. Time Elapse = {0}'.format(timeit.timeit("functionQ3d(k = " + str(k) + ",N = 20000)", setup="from __main__ import functionQ3d", number = 1)))
    print('Matrix Method for 3 ground state. Time Elapse = {0}'.format(timeit.timeit("functionQ3d1(k = " + str(k) + ",N = 310000)", setup="from __main__ import functionQ3d1", number = 1)))


    

# Intuitively, it seems that solving the equation with matrix finite difference equation provide the most straightforward result
# as it compute the value of the wavefunction/the solution of the differential equation for all coordinate in a second of time for all coordinates.
# Personally speaking, for higher degree of accuracy achieve by implementing the numerov or other computational ODE solver instead of matrix.
# From benchmark 4, it seems that the result with N = 320000 provide accuracy smaller than the numerov
# Yet, the result obtained for test case ground state and may differ for higher state
# It seems the numerov algorithm gradually fails in faster rate for higher eigenstate compared to the matrix method
# This means we are working with 320000-by-320000 square matrix.
# It can get computationally expensive with larger N and doesn't provide very precise result.