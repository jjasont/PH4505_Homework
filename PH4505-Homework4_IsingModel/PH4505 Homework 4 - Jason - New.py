# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:30:23 2018
2D Ising Model
Question 1 (20/20)
Good job!
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
    [rows_i, cols_j] = arr.shape
    for i in range(0,rows_i*cols_j): #remove un-indent any code under the following line if one spin per MC desired
        arr1 = arr
        [rows_i, cols_j] = arr.shape
        chosen_i = np.random.randint(0,rows_i) # from 0 to rows-1
        chosen_j = np.random.randint(0,cols_j) # from 0 to cols-1
        chosen_spin = arr[chosen_i, chosen_j]
    
            
        E = calculateLocalEnergy(arr, chosen_i, chosen_j, nbor)
    
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

    energy *= -J/(rows*cols)
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
    #print('Built-In Function mag: Mean = {0}, Var = {1}'.format(np.mean(mag),np.var(mag, ddof = 1)))
    mag_mean = np.sum(mag)/len(mag)
    mag_var = np.sum((mag - np.mean(mag))**2)/(len(mag)-1)#np.sum(mag2 - mag_mean**2)/(len(mag)-1) #np.sum((mag - np.mean(mag))**2)/(len(mag)-1) # this is variance
    print('Calculate mag: Mean = {0}, Var = {1}'.format(mag_mean,mag_var))
    enr_mean = np.sum(enr)/len(enr)
    enr_var = np.sum((enr - np.mean(enr))**2)/(len(enr)-1)#np.sum(enr2 - enr_mean**2)/(len(enr)-1) 
    #print('Built-In Function enr: enr = {0}, Var = {1}'.format(np.mean(enr),np.var(enr, ddof = 1)))
    print('Calculate enr: Mean = {0}, Var = {1}'.format(enr_mean,enr_var))
    return mag_mean, mag_var, enr_mean, enr_var

def average2(L, enr, enr2, mag, mag2, J, T):
    cv = ((enr2 - enr**2)/(L**2 * T**2))
    cv_mean = np.sum(cv)/len(enr)
    cv_var = np.sum((cv-cv_mean)**2) / (len(enr) - 1)
    #print('Built-In Function specific_heat: enr = {0}, Var = {1}'.format(np.mean(specific_heat),np.var(specific_heat, ddof = 1)))
    print('Calculate cv: Mean = {0}, Var = {1}'.format(cv_mean,cv_var))
    sus = (mag2-mag**2) * L**2 / T
    sus_mean = np.sum(sus)/len(mag)
    sus_var = np.sum((sus - sus_mean)**2) / (len(mag) - 1)
    print('Calculate magnet_sus: Mean = {0}, Var = {1}'.format(sus_mean,sus_var))
    return cv_mean, cv_var, sus_mean, sus_var
    
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
                enr[bin_id] = np.mean(be) #total energy
                enr2[bin_id] = np.mean(be2) #total energy^2
                mag[bin_id] = np.mean(np.abs(bm))
                mag2[bin_id] = np.mean(bm2)
                be,bm,be2,bm2 = [], [], [], []
                print('The {0}-th bin calculated'.format(bin_id+1))
            else:
                bm.append(m)
                bm2.append(m2)
                be.append(e)
                be2.append(e2)
    print('............................................')
    mag_mean, mag_var, enr_mean, enr_var = average1(enr, enr2, mag, mag2)
    cv_mean, cv_var, sus_mean, sus_var = average2(L, enr, enr2, mag, mag2, J, T)
    #print(sus_mean)    
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
    
    return mag_mean, mag_var, enr_mean, enr_var, cv_mean, cv_var, sus_mean, sus_var, mag_time

# Iteration for temperatue between 0.1 and 4.0 approximately follow normal distribution
# with mean 2.269 for ~256 temperature
#nt = 25#2**8
#trial_T = np.random.normal(2.269, .90, nt)
#trial_T[trial_T == np.min(trial_T)] = 0.1
#trial_T = trial_T[trial_T >= 0.1]
#trial_T[trial_T == np.max(trial_T)] = 4.0
#trial_T = trial_T[trial_T <= 4.0]
#trial_T = np.sort(trial_T)
#trial_T = [0.1]
trial_T = np.linspace(0.1,4.0,40)
#lattice_size = [4, 8, 16]

#for l in lattice_size:
# Storing for each temperature
#enr_T = []
#enr_Terr = []
#mag_T = []
#mag_Terr = []
#specifich_T = []
#specifich_Terr = []
#magsuscp_T = []
#magsuscp_Terr = []

# Storing for each temperature and lattice
enr_TL = []
enr_TLerr = []
mag_TL = []
mag_TLerr = []
specifich_TL = []
specifich_TLerr = []
magsuscp_TL = []
magsuscp_TLerr = []
trial_L = [4,8,16]


fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots( nrows=2, ncols=2 , figsize = [15,15], sharex = True)

for l, linear_i in zip(trial_L, range(0,len(trial_L))):
    print('Value of L = {0}'.format(l))
    enr_T = []
    enr_Terr = []
    mag_T = []
    mag_Terr = []
    specifich_T = []
    specifich_Terr = []
    magsuscp_T = []
    magsuscp_Terr = []
    for t, temperature_i in zip(trial_T,range(1,len(trial_T)+1)):
        print('The {0}-th temperature'.format(temperature_i))
        mag_mean, mag_var, enr_mean, enr_var, cv_mean, cv_var, sus_mean, sus_var, mag_time = simulation(mstep = 10000, nequil = 10000, nbins = 5, L = 10, T = t, verbose = False)
        enr_T.append(enr_mean)
        enr_Terr.append(np.sqrt(enr_var))
        mag_T.append(mag_mean)
        mag_Terr.append(np.sqrt(mag_var))
        specifich_T.append(cv_mean)
        specifich_Terr.append(np.sqrt(cv_var))
        magsuscp_T.append(sus_mean)
        magsuscp_Terr.append(np.sqrt(sus_var))
   
    enr_TL.append(enr_T)
    enr_TLerr.append(enr_Terr)
    mag_TL.append(mag_T)
    mag_TLerr.append(mag_Terr)
    specifich_TL.append(specifich_T)
    specifich_TLerr.append(specifich_Terr)
    magsuscp_TL.append(magsuscp_T)
    magsuscp_TLerr.append(magsuscp_Terr)
        
    ax11.errorbar(trial_T, mag_TL[linear_i],yerr = mag_TLerr[linear_i], marker = 's', label = 'L = ' + str(l),linestyle = 'None')
    ax11.set_xlabel('Temperature')
    ax11.set_ylabel('Average Magnetization')
    ax11.set_title('Average Magnetization per site vs T')
    ax11.legend()
    
    ax12.errorbar(trial_T, enr_TL[linear_i],yerr = enr_TLerr[linear_i], marker = 's', label = 'L = ' + str(l),linestyle = 'None')
    ax12.set_xlabel('Temperature')
    ax12.set_ylabel('Average Energy/Site')
    ax12.set_title('Average Energy per Site vs T')
    ax12.legend()
    
    ax21.errorbar(trial_T, specifich_TL[linear_i], yerr = specifich_TLerr[linear_i], marker = 's', label = 'L = ' + str(l),linestyle = 'None')
    ax21.set_xlabel('Temperature')
    ax21.set_ylabel('Specific Heat, Cv')
    ax21.set_title('Specific Heat, Cv vs T')
    ax21.legend()
    
    ax22.errorbar(trial_T, magsuscp_TL[linear_i], yerr = magsuscp_TLerr[linear_i], marker = 's', label = 'L = ' + str(l) ,linestyle = 'None')
    ax22.set_xlabel('Temperature')
    ax22.set_ylabel('Magnetic Susceptibility, Xm')
    ax22.set_title('Magnetic Susceptibility, Xm vs T')
    ax22.legend()
    
fig.tight_layout()
#fig.savefig('D:\OneDrive\OneDrive - Nanyang Technological University\Dropbox\PHMA Stuff\Year 4\Year 4 Sem 2\PH4505 Computational Physics\PH4505 Homework\\result2-test10000_7.png')
#np.savez('D:\OneDrive\OneDrive - Nanyang Technological University\Dropbox\PHMA Stuff\Year 4\Year 4 Sem 2\PH4505 Computational Physics\PH4505 Homework\\nequil = 10000, mstep = 1000, nbins = 10, L = 16\\result1',trial_T = trial_T, mag_T = mag_T, mag_Terr = mag_Terr, enr_T = enr_T, enr_Terr = enr_Terr, specifich_T = specifich_T, specifich_Terr = specifich_Terr, magsuscp_T = magsuscp_T, magsuscp_Terr = magsuscp_Terr)

#np.savez('D:\OneDrive\OneDrive - Nanyang Technological University\Dropbox\PHMA Stuff\Year 4\Year 4 Sem 2\PH4505 Computational Physics\PH4505 Homework\\nequil = 1000, mstep = 1000, nbins = 5, L = 4,8,16\\result1',trial_L = trial_L,
#trial_T = trial_T,
#enr_TL = enr_TL,
#enr_TLerr = enr_TLerr,
#mag_TL = mag_TL,
#mag_TLerr = mag_TLerr,
#specifich_TL = specifich_TL,
#specifich_TLerr = specifich_TLerr,
#magsuscp_TL = magsuscp_TL,
#magsuscp_TLerr = magsuscp_TLerr)

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


# - Some point in the plot at low T seems to be off from the majority, this caused that the algorithm reach an equilibirum which
#   not all the spin pointing the same direction (meta-stable cluster) i.e. create a band cluster
# - This is the limitation of Metropolis algorithm for low temperature.


#%% ------------------------- USED FOR LOADING SAVE DATA AND PLOT -----------------------------------
#result = np.load('D:\OneDrive\OneDrive - Nanyang Technological University\Dropbox\PHMA Stuff\Year 4\Year 4 Sem 2\PH4505 Computational Physics\PH4505 Homework\nequil = 10000, mstep = 10000, nbins = 10, L = 16\\result.npz')
#trial_T = result['trial_T']
#mag_T = result['mag_T']
#mag_Terr = result['mag_Terr']
#enr_T = result['enr_T']
#enr_Terr = result['enr_Terr']
#specifich_T = result['specifich_T']
#specifich_Terr = result['specifich_Terr']
#magsuscp_T = result['magsuscp_T']
#magsuscp_Terr = result['magsuscp_Terr']
#
#fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots( nrows=2, ncols=2 , figsize = [15,15], sharex = True)
#ax11.errorbar(trial_T, mag_T,yerr = mag_Terr, marker = 's', color = 'blue', linestyle = 'None')
#ax11.set_xlabel('Temperature')
#ax11.set_ylabel('Average Magnetization')
#ax11.set_title('Average Magnetization vs T')
#
#ax12.errorbar(trial_T, enr_T,yerr = enr_Terr, marker = 's', color = 'red', linestyle = 'None')
#ax12.set_xlabel('Temperature')
#ax12.set_ylabel('Average Energy')
#ax12.set_title('Average Energy vs T')
#
#ax21.errorbar(trial_T, specifich_T, yerr = specifich_Terr, marker = 's', color = 'black', linestyle = 'None')
#ax21.set_xlabel('Temperature')
#ax21.set_ylabel('Specific Heat, Cv')
#ax21.set_title('Specific Heat, Cv vs T')
#
#ax22.errorbar(trial_T, magsuscp_T, yerr = magsuscp_Terr, marker = 's', color = 'green', linestyle = 'None')
#ax22.set_xlabel('Temperature')
#ax22.set_ylabel('Magnetic Susceptibility, Xm')
#ax22.set_title('Magnetic Susceptibility, Xm vs T')

import matplotlib.pyplot as plt

trial_T = result['trial_T']
mag_T = result['mag_T']
mag_Terr = result['mag_Terr']
enr_T = result['enr_T']
enr_Terr = result['enr_Terr']
specifich_T = result['specifich_T']
#specifich_Terr = result['specifich_Terr']
magsuscp_T = result['magsuscp_T']
#magsuscp_Terr = result['magsuscp_Terr']


fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots( nrows=2, ncols=2 , figsize = [15,15], sharex = True)
ax11.errorbar(trial_T, mag_T,yerr = mag_Terr, marker = 's', color = 'blue', linestyle = 'None')
ax11.set_xlabel('Temperature')
ax11.set_ylabel('Average Magnetization')
ax11.set_title('Average Magnetization vs T')

ax12.errorbar(trial_T, enr_T/256,yerr = enr_Terr/256, marker = 's', color = 'red', linestyle = 'None')
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

