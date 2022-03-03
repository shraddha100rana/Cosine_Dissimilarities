import numpy as np
import scipy.stats

def MoransI(Z, W):
    #Z is a n x 1 matrix of values at nodes minus the mean of values at all nodes
    #W is the n x n weight matrix between node i and j
    N = len(Z)
    S0 = sum(W.sum(axis = 0))
    #index value, calculated value of I
    I = (N*(np.matmul(np.matmul(Z.transpose(),W),Z)))/(S0*((Z**2).sum(axis=0)))
    I = float(I)
    
    #expected value of I
    EI = -1/(N-1)
    
    S2 = sum((W.sum(axis=0)+W.sum(axis=1))**2)
    S1 = 0.5*sum(np.square(W+W.transpose()).sum(axis=0))
    D = ((Z**4).sum(axis = 0))/(((Z**2).sum(axis=0))**2)
    D = float(D)
    C = (S0**2)*(N-1)*(N-2)*(N-3)
    B = D*(S1*((N**2)-N)-(2*N*S2)+6*(S0**2))
    A = N*(S1*((N**2)-3*N+3)-(N*S2)+3*(S0**2))
    #expected value of I^2
    EI2 = (A-B)/C
    
    #variance of I
    VI = EI2-(EI**2)
    
    #z-score
    Z_score = (I-EI)/np.sqrt(VI)
    
    #p-value from z-score
    p_value = scipy.stats.norm.sf(abs(Z_score))*2
    
    return (I, Z_score, p_value)
