import scipy

def get_pi(P):
    eig = scipy.linalg.eig(P,left=True,right=False)[1][:,0]
    return np.real(eig/sum(eig))

