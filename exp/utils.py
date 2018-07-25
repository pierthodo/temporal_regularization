import scipy.linalg as la
import numpy as np

def get_pi(P):
    eig = la.eig(P,left=True,right=False)[1][:,0]
    return np.real(eig/sum(eig))

