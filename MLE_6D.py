#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
import matplotlib.pylab as plt
import time
from scipy import optimize
# from numba import jit
import pandas as pd

Dim = 6 # Dimensionality of the system
nDrift = 28 # terms for each Dimension of the drift
nDiff = 1 # terms for the Diffusion coefficient



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
    sigma = sigma[1:] # without noise parameters
    sigma=sigma.reshape((Dim,-1))
    function=np.zeros((len(x),Dim))
    for i in range(Dim):
        function[:,i]=poly(x.T,sigma[i])
    return function



#@jit
def D2(alpha,x):
    return np.outer(np.eye(Dim,Dim),(alpha[0])).reshape((Dim,Dim,-1)) # Noise = alpha[0]


def det_D2(alpha,x):
    return (alpha[0]*np.ones(x.shape[0]))**Dim


def inv_D2(alpha,x):
    return np.outer(np.eye(Dim,Dim),
                    1/(alpha[0]*np.ones(x.shape[0]) )).reshape((Dim,Dim,-1))


#  Log Likelihood and negative logL

def log_likelihood(alpha,x,dt):
    # alpha is the current set of parameters
    # x is the entire data set N x 2
    # dt is the time difference
    
    log_like = 0 # initial value of sum
    
    #calculate D1 and D2 or each position in the data x
    
    if alpha[0]>0: # noise must be positive!
        
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


def neg_log_likelihood(alpha,x,dt): #L Threshold Lambdac
    return -1*log_likelihood(alpha,x,dt)


# Log_Likelihood for after some parameters were cut off by the Threshold

def second_neg_log_likelihood(Coeff, Index,x,dt):
    # Index: Index of those coefficients which are set to 0: Boolean 
    Index[0] = False # noise NEVER cut off
    Coeff[Index] = 0
    return -1*log_likelihood(Coeff,x,dt)



# BIC as goodness criterion for a threshold value

def BIC(alpha,x,dt,L): # mit Lambda Threshold
    
    logi = np.abs(alpha)>L # which are larger than Lambda?
    logi[0] = True  # noise is always included
    return np.log(x[:,0].size)*np.sum(  logi ) - 2*log_likelihood(alpha, x,dt )


# Calculate BIC in the Loop with thresholding

def Loop(x, dt, L, a_Ini):
    # estimates alpha parameters based on starting values a_Ini for a given threshold L
    a_hat = optimize.minimize(neg_log_likelihood, a_Ini,args=(x,dt)) # max likelihood
    
    for i in np.arange(0,n_Cut):
        Cut = (np.abs(a_hat["x"])<L) # Boolean of the values that are cut off
        # second optimization with maxLikelEstimator as start:
        a_hat = optimize.minimize(second_neg_log_likelihood,a_hat["x"],args = (Cut,x,dt))
    return(a_hat["x"])




#Get Data

filename = 'CoupledLorenz_dt0.01' 
x = np.load(filename+'.npy')
dx = x[1:]-x[:-1]  
dt = float(filename[-4:])  #time step in file name
Lambda = 0.5 # initial threshold
print(x.shape)
N = x[:,0].size



# Set up variables for the hyperparameter search on threshold

n_Cut = 5 # Number of reiterating
hp1 = np.arange(0.00, 1.2, 0.25) # list of possible thresholds
n_Iteration = len(hp1) # Number of Hyperparameter search iterations
score = np.empty(n_Iteration) # score for Hyperparameters


# In[5]:


TestAl = np.ones(nDiff + Dim*nDrift)   # sample parameters to start the search
AlphaList = np.empty((n_Iteration, TestAl.size)) # store the results of each optimization here 


# In[6]:
             

# comparison: the true parameters
real = np.zeros(nDiff + Dim*nDrift)
real[0]= 2 #noise1

# x1 component
real[2] = -10. # x in x
real[3] = 10.  # y in x

# y1 component
real[30] = 28. # x in y
real[31] = -1. # y in y
real[38] = -1  # xz in y

# z1 component
real[60] = -8./3. # z in z
real[65] = 1. # xy in z
                    
# x2 component
real[89] = -10.+ 1. # x in x
real[90] = 10.  # y in x
real[86] = -1 # coupling of the other system

# y2 component
real[117] = 28. # x in y
real[118] = -1. # y in y
real[137] = -1  # xz in y

# z2 component
real[147] = -8./3. # z in z
real[164] = 1. # xy in z

# to show the important terms
output = (real!=0) #relevant terms



TestAl = optimize.minimize(neg_log_likelihood, 
                           np.ones(nDiff + Dim*nDrift),
                           args=(x[0:1000,:],   # first estimation, maybe with little data
                                 dt)) 

TestAl = TestAl["x"]

print(TestAl)

# In[ ]:

print("Real:",real[output])

print("Estimated:", TestAl[output])

for i in range(n_Iteration):
    estimate = Loop(x, dt, hp1[i], TestAl)
    AlphaList[i,:] = estimate
    score[i] = BIC(estimate, x, dt, hp1[i])
    print(estimate[output])
    print(hp1[i], score[i])


np.savetxt('CoupledLorenz_Alpha_Done.txt',AlphaList)
d = {'Threshold': hp1, 'BIC': score}
df = pd.DataFrame(data=d)
print(df)
df.to_csv("Lorenz_Score_BIC.csv")