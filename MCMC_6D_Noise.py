#!/usr/bin/env python
# coding: utf-8

# Assume we already have an alright estimate for the Drift coefficients
# Use the drift coefficients to estimate the dynamic noise effects

import emcee
import numpy as np
import matplotlib.pylab as plt
import time
from scipy import optimize
# from numba import jit
import pandas as pd

Dim = 6 # Dimensionality of the system
nDrift = 28 # terms for each Dimension of the drift
nDiff = 1 + 2*Dim #+Dim # terms for the Diffusion coefficient


#Get Data
filename = 'CoupledLorenz_dt0.01' 
x = np.load(filename+'.npy')
dx = x[1:]-x[:-1]  
dt = float(filename[-4:])  #time step in file name
Lambda = 0.5 # initial threshold
print(x.shape)
N = x[:,0].size


### Take old drift estimation into account
Alphas = np.loadtxt('CoupledLorenz_Alpha_Done.txt')
df0 = pd.read_csv("Lorenz_Score_BIC.csv")
BestThreshold = df0.loc[df0['BIC'].idxmin()]['Threshold']
BestCoeff = Alphas[df0["Threshold"] == BestThreshold,:]
# BestCoeff[0][1:] are the drift coefficients from the 1st estimation
# BestCoeff[0][0] is the constant noise estimate
 
    
# let's only regard up to second order    
def poly(x,sigma):
    x_vec=np.array([1,x[0],x[1],x[2],x[3],x[4],x[5],  #7
                   x[0]**2.,  x[0]*x[1], x[0]*x[2], x[0]*x[3], x[0]*x[4], x[0]*x[5],# 6 terms
                    x[1]**2.,x[1]*x[2], x[1]*x[3], x[1]*x[4], x[1]*x[5], # 5 terms
                    x[2]**2.,  x[2]*x[3], x[2]*x[4], x[2]*x[5], # 4 terms
                    x[3]**2., x[3]*x[4], x[3]*x[5], x[4]**2., x[4]*x[5], x[5]**2]) # 6 terms
    return np.dot(sigma,x_vec) # Total: 28 terms

#@jit
def D1(sigma,x):
    sigma = sigma[nDiff:] # without noise parameters
    sigma=sigma.reshape((Dim,-1))
    function=np.zeros((len(x),Dim))
    for i in range(Dim):
        function[:,i]=poly(x.T,sigma[i])
    return function

def DiffTerm(alpha,x):
    x_vec = np.array([np.ones(x.shape[0])**2,
                      np.abs(x[:,0]), np.abs(x[:,1]), np.abs(x[:,2]), np.abs(x[:,3]), np.abs(x[:,4]), np.abs(x[:,5]),
                      x[:,0]**2, x[:,1]**2,x[:,2]**2, x[:,3]**2, x[:,4]**2, x[:,5]**2])
    return(np.dot(alpha[0:nDiff], x_vec))
    #return(alpha[0])

#@jit
def D2(alpha,x):
    return np.outer(np.eye(Dim,Dim),DiffTerm(alpha,x)).reshape((Dim,Dim,-1)) # Noise = alpha[0:nDiff]


def det_D2(alpha,x):
    return (DiffTerm(alpha,x)*np.ones(x.shape[0]))**Dim


def inv_D2(alpha,x):
    return np.outer(np.eye(Dim,Dim),
                    1/(DiffTerm(alpha,x)*np.ones(x.shape[0]) )).reshape((Dim,Dim,-1))


#  Log Likelihood and negative logL

def log_likelihood(alpha_in,x,dt):
    # alpha is the current set of parameters
    # x is the entire data set N x 2
    # dt is the time difference
    
    # HERE: alpha-input are only the diffusion terms
    # combine them with the known drift terms from 1st estimation
    alpha = np.concatenate((alpha_in, BestCoeff[0][1:]))
    
    log_like = 0 # initial value of sum
    
    #calculate D1 and D2 or each position in the data x
    
    if max(alpha[0:nDiff])>0: # some noise term must be positive!
        
        dx = x[1:,:]-x[:-1,:]  
        
        d1 = dx -D1(alpha,x)[:-1,:]*dt
        d2 = D2(alpha,x)
        d2_inv = inv_D2(alpha,x)
        d2_det = det_D2(alpha,x)[:-1]
        

        r = np.array([np.dot(d1[i,:],
                             np.dot(d2_inv[:,:,i].T,
                                    d1[i,:])) for i in range(len(x)-1)])

        # HERE: Instead of summing all components (i.e. every time step), 
        #       one could just sum over a small subset to increase computation speed?
        #print((-r/(2*dt)-np.log(np.sqrt(4*np.pi*dt)**Dim*np.sqrt(d2_det))).shape)
        #log_like = (-r/(2*dt)-np.log(np.sqrt(4*np.pi*dt)**Dim*np.sqrt(d2_det))).sum()
        log_like = (-r/(2*dt)-np.log(np.sqrt(4*np.pi*dt)**Dim*np.sqrt(d2_det)))
        log_like = log_like.sum()
        return log_like
    else:
        return -np.inf



    
##############################
## Now for the MCMC part #####

nWalkers = 1000
Start = np.random.rand(nWalkers,
                       nDiff) # only for diffusion parameters


sampler = emcee.EnsembleSampler(nWalkers,nDiff,log_likelihood, args = (x,dt))

print("Burn-In begins")
# burn-in: hopefully enough!
state = sampler.run_mcmc(Start, 100) 
sampler.reset()

print("Sampling begins")
# Now the real sampling
sampler.run_mcmc(state, 1500)
samples = sampler.get_chain(flat=True, discard = 500) #discard: more burn-in
np.savetxt("SampleDoubleLorenz.txt",samples)
