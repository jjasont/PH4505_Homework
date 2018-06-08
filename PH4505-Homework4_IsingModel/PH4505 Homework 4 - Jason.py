# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:30:23 2018
2D Ising Model
@author: Jason Tanuwijaya
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
nbor = [[-1,0], [1,0], [0,-1], [0,1]] #[increase for i, increase for j]

# calculateLocalEnergy is a function with following parameter
# arr - 2D array configuration of Ising Model
# chosen_i, chosen_j - chosen site index i,j of the array for energy calculation
# nbor - list of nearest neighborhood
#		 for calculation of probability use 'nbor' (von Neumann Neighborhood)
#	 	 for calculation of total energy use 'nbor1' (neighbor on the bottom and right) to avoid double count

def calculateLocalEnergy(arr, chosen_i, chosen_j, nbor):
    local_energy = 0
    [rows_i, cols_j] = arr.shape
    chosen_site = np.array([chosen_i, chosen_j])
    chosen_spin = arr[chosen_i, chosen_j]
    for bor in nbor:
        neighbor = chosen_site+np.array(bor)
        index = neighbor == len(arr) #if any of the neighbor on the boundary (size of the array)
        if np.any(index):
            neighbor = neighbor%len(arr) #negative index won't be affected
            local_energy += chosen_spin*arr[neighbor[0],neighbor[1]] #add the energy of with new neighborhood index
        else:
            local_energy += chosen_spin*arr[chosen_i + bor[0], chosen_j + bor[1]]
    
    return local_energy

def metropolis(arr, J, T, pacc, i):
#    kT = 5#20
#    J = 1
    arr1 = arr
    [rows_i, cols_j] = arr.shape
    chosen_i = np.random.randint(0,rows_i) # from 0 to rows-1
    chosen_j = np.random.randint(0,cols_j) # from 0 to cols-1
    #print('CHosen site is x = {0} and y = {1}'.format(chosen_x, chosen_y))
#    E0 = 0
#    chosen_site = np.array([chosen_i, chosen_j])
#    #print(chosen_site)
    chosen_spin = arr[chosen_i, chosen_j]
#    #print('============E0==============')
#    for bor in nbor:
#        neighbor = chosen_site+np.array(bor)
#        index = neighbor == len(arr)
#        if np.any(index):
#            #print('Boundary')
#            neighbor = neighbor%len(arr)
#            E0 += chosen_spin*arr[neighbor[0],neighbor[1]]
#        else:
#            #print('Non-Boundary')
#            E0 += chosen_spin*arr[chosen_i + bor[0], chosen_j + bor[1]]
        
    E = calculateLocalEnergy(arr, chosen_i, chosen_j, nbor)
        #np.any(np.array([chosen_i, chosen_j])+np.array(bor) == len(arr))
        
    #print('E0 is {0}'.format(E0))
        
 
    
    #print('============--==============')
#    E1 = 0
#    
#    #chosen_spin_flip = flip_spin
#    for bor in nbor:
#        neighbor = chosen_site+np.array(bor)
#        index = neighbor == len(arr)
#        if np.any(index):
##            print('Boundary')
#            E1 += flip_spin*arr[(neighbor%len(arr)).tolist()]
#        else:
##            print('Non-Boundary')
#            E1 += flip_spin*arr[chosen_i + bor[0], chosen_j + bor[1]]
#    print('E1 is {0}'.format(E1))
#    print('============E1==============')
#    for bor in nbor:
#        if (chosen_y+bor[0])%rows == 0 & (chosen_x+bor[1])%cols == 0:
#            E1 +=flip_spin + arr[0, 0]
#        elif (chosen_y+bor[0]+1)%rows == 0:
#            print([chosen_y, chosen_x] + bor)
#            print('============1==============')
#            E1 += flip_spin + arr[0, chosen_x+bor[1]]
#        elif (chosen_x+bor[1]+1)%cols == 0:
#            print([chosen_y, chosen_x] + bor)
#            print('============2==============')
#            E1 += flip_spin + arr[chosen_y+bor[0], 0]
#        else:
#            print([chosen_y, chosen_x] + bor)
#            print('============3==============')
#            E1+= flip_spin+ arr[chosen_y+bor[0], chosen_x+bor[1]]
    
    #print('============--==============')
    #w0 = pacc.get(int(E0))
    #w1 = pacc.get(int(E1))
#    e = E1 - E0
    
   # flip_spin = -arr[chosen_i, chosen_j] 
#    print('e difference is {0}'.format(e))
#    if E1 < E0:
#        #print('Initial energy = {0}, After flip energy = {1}'.format(E0, E1))
#        arr1[chosen_i, chosen_j]  = flip_spin
#    elif E1 > E0:
        #print('Initial energy = {0}, After flip energy = {1}'.format(E0, E1))
        #print("W' /W = {0}".format(np.exp(-J*(E1-E0)/kT)))
    if pacc.get(int(E)) > np.random.random_sample():
        chosen_spin *= -1
        
    arr1[chosen_i, chosen_j]  = chosen_spin
            #print('spin flip')
    
    m, m2, energy, energy2 = measureParameter(arr1, J)
    
    return arr1, m, m2, energy, energy2

def measureParameter(arr, J):
    m = np.mean(arr)
    m2 = m**2
    [rows, cols] = arr.shape
    
    x, y = np.meshgrid(np.arange(0,len(arr)),np.arange(0,len(arr)))
    energy = 0
    #avoid double count. only calculate energy for botom neighbor and right neighbor
    nbor1 = [nbor[1], nbor[3]] 
    for j,i in zip(x.flatten(),y.flatten()):
        chosen_site = np.array([i, j])
        energy += calculateLocalEnergy(arr, chosen_site[0], chosen_site[1], nbor1)

    energy *= -J
    energy2 = energy**2
    return m, m2, energy, energy2

def createPacc(J, T):
    pacc = []
    for e_diff in np.linspace(-4,4,5):
        # Positive J is ferromagnetic
        # Negative J is anti-ferromagnetic
        pacc.append((int(e_diff), np.exp(-2*J*e_diff/T)))  
    return dict(pacc)

def average1(enr, enr2, mag, mag2):
    print('Built-In Function mag: Mean = {0}, Var = {1}'.format(np.mean(mag),np.var(mag, ddof = 1)))
    mag_mean = np.sum(mag)/len(mag)
    mag_var = np.sum((mag - np.mean(mag))**2)/(len(mag)-1)#np.sum(mag2 - mag_mean**2)/(len(mag)-1) #np.sum((mag - np.mean(mag))**2)/(len(mag)-1) # this is variance
    print('Calculate mag: Mean = {0}, Var = {1}'.format(mag_mean,mag_var))
    enr_mean = np.sum(enr)/len(enr)
    enr_var = np.sum((enr - np.mean(enr))**2)/(len(enr)-1)#np.sum(enr2 - enr_mean**2)/(len(enr)-1) 
    print('Built-In Function enr: enr = {0}, Var = {1}'.format(np.mean(enr),np.var(enr, ddof = 1)))
    print('Calculate enr: Mean = {0}, Var = {1}'.format(enr_mean,enr_var))
    return mag_mean, mag_var, enr_mean, enr_var

def average2(enr, enr2, J, T):
    specific_heat = len(enr)*(np.mean(enr2) - np.mean(enr)**2)
    return specific_heat
    
def simulation(mstep = 10000, nequil = 10000, nbins = 10, L = 16, J = 1, T = 1, verbose = False):
    size = [L, L]
    init_down = 0.5 #Fraction of down spin
    spn = np.random.choice([-1,1],size, p =[init_down, 1-init_down])


    enr = np.zeros(nbins)
    mag = np.zeros(nbins)
    enr2 = np.zeros(nbins)
    mag2 = np.zeros(nbins)
    mag_time = np.zeros(nequil+mstep*nbins)
    be,bm,be2,bm2 = [], [], [], []
    print('============Simulation Parameter===========')
    print('~ mstep = {0}'.format(mstep))
    print('~ nequil = {0}'.format(nequil))
    print('~ nbins = {0}'.format(nbins))
    print('~ Size = {0}x{1}'.format(L,L))
    print('~ Coupling constant, J = {0}'.format(J))
    print('~ Temperature, T = {0}'.format(T))
    print('================================')
    pacc = createPacc(J, T)
    
    if verbose:
    # Shows the original configuration
        plt.imshow(spn)
        plt.title('Initial Configuration')
        plt.grid(False)
        plt.show()
    
    
    for iteration in range(0, nequil+mstep*nbins):
        spn, m, m2, e, e2 = metropolis(spn, J, T, pacc, iteration)
        bm.append(m)
        bm2.append(m2)
        be.append(e)
        be2.append(e2)
        mag_time[iteration] = m
        
        if (iteration+1)%(5000) == 0:
            print('Iteration {0}'.format(iteration+1))
        if (iteration+1 - nequil) == 0:
            print('Equilibration Step Completed')
            print('-----------------------------------------')
            continue
        elif (iteration+1 - nequil) < 0:
            continue
        else:
            if (iteration+1 - nequil)%mstep == 0:
                bin_id = int((iteration+1 - nequil)/mstep - 1)
                enr[bin_id] = np.mean(be)
                enr2[bin_id] = np.mean(be2)
                mag[bin_id] = np.mean(np.abs(bm))
                mag2[bin_id] = np.mean(bm2)
                be,bm,be2,bm2 = [], [], [], []
                print('The {0}-th bin calculated'.format(bin_id+1))
                print('............................................')
            else:
                bm.append(m)
                bm2.append(m2)
                be.append(e)
                be2.append(e2)
    mag_mean, mag_var, enr_mean, enr_var = average1(enr, enr2, mag, mag2)
    specific_heat = average2(enr, enr2, J, T)
    mag_suscp = np.mean(mag2) - np.mean(mag)**2
    
    print('The specific heat Cv = {0} at T = {1} and J = {2}'.format(specific_heat, T, J))
    print('The magnetic susceptibility Xm = {0} at T = {1} and J = {2}'.format(mag_suscp, T, J))
    
    if verbose:
    # Plot magnetization across time
        plt.plot(mag)
        plt.show()
        plt.plot(mag_time)
        plt.show()
    # Shows the final configuration
        plt.imshow(spn)
        plt.title('Final Configuration')
        plt.grid(False)
        plt.show()
    
    return enr, enr2, mag, mag2, mag_mean, mag_var, enr_mean, enr_var, specific_heat, mag_suscp, mag_time

# Iteration for temperatue between 0.1 and 4.0 approximately follow normal distribution
# with mean 2.269 for ~256 temperature
nt = 2**8

trial_T = np.random.normal(2.269, .90, nt)
trial_T[trial_T == np.min(trial_T)] = 0.1
trial_T = trial_T[trial_T >= 0.1]
trial_T[trial_T == np.max(trial_T)] = 4.0
trial_T = trial_T[trial_T <= 4.0]
trial_T = np.sort(trial_T)
trial_T = [0.1]

enr_T = []
enr_Terr = []
mag_T = []
mag_Terr = []
specifich_T = []
magsuscp_T = []

for t in trial_T:
    enr, enr2, mag, mag2, mag_mean, mag_var, enr_mean, enr_var, specific_heat, mag_suscp, mag_time = simulation(mstep = 100, nequil = 5000, nbins = 10, L = 15, T = t, verbose = True)
    enr_T.append(enr_mean)
    enr_Terr.append(np.sqrt(enr_var))
    mag_T.append(mag_mean)
    mag_Terr.append(np.sqrt(mag_var))
    specifich_T.append(specific_heat)
    magsuscp_T.append(mag_suscp)
    
fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots( nrows=2, ncols=2 , figsize = [15,15], sharex = True)
ax11.plot(trial_T, mag_T,'bo')
ax11.set_xlabel('Temperature')
ax11.set_ylabel('Average Magnetization')
ax11.set_title('Average Magnetization vs T')

ax12.plot(trial_T, enr_T,'ro')
ax12.set_xlabel('Temperature')
ax12.set_ylabel('Average Energy')
ax12.set_title('Average Energy vs T')

ax21.plot(trial_T, specifich_T,'ko')
ax21.set_xlabel('Temperature')
ax21.set_ylabel('Specific Heat, Cv')
ax21.set_title('Specific Heat, Cv vs T')

ax22.plot(trial_T, magsuscp_T,'go')
ax22.set_xlabel('Temperature')
ax22.set_ylabel('Magnetic Susceptibility, Xm')
ax22.set_title('Magnetic Susceptibility, Xm vs T')

fig.tight_layout()
#fig.savefig('D:\OneDrive\OneDrive - Nanyang Technological University\Dropbox\PHMA Stuff\Year 4\Year 4 Sem 2\PH4505 Computational Physics\PH4505 Homework\\result2-test5000_3.png')
#plt.close(fig)

# The critical temperature of the Ising model found to be somewhere
# around 2.3 and 2.5
# Magnetization
# - At low T, it have high magnetization (the absolute maximum is 1) as it's in their ordered state and
#   all of the spin aligned mostly in the same direction
# - At high T, it have almost 0 magnetization as the system is unordered
#   at have random configuration
# Energy
# - At low T, it have low energy and gradually increase with higher temperature. Again, similar explanation for out of place point
# Magnetization & specific heat
# - At low and high T, it has low value. As the temperature approaching the critical temperature, the value gradually increase and diverge forming a discontinuity


# - Some point in the plot at low T seems to of from the majority, this caused that the algorithm reach an equilibirum which
#   not all the spin pointing the same direction i.e. create a band cluster
