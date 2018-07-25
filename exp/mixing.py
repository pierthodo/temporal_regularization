from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib
from utils import get_pi

def mixing():
    size = 10
    T = 5
    rep_list = np.empty((100,500,3))
    for rep in range(100):
        matrix = np.random.random(size=(size,size)).astype(np.float64)
        matrix = matrix/matrix.sum(axis=1)[:,None]
        pi = get_pi(matrix)
        matrix_s = np.diag(pi**(-1)).dot(matrix.T).dot(np.diag(pi))
        result = np.empty((100,500,3))
        for n in range(100):
            dist = []
            for b in range(100):
                for i in range(T):
                    m = (1-b/100)*matrix + (b/100)*matrix_s
                    dist.append([b/100,i,np.linalg.norm((np.ones(size)/size).dot(np.linalg.matrix_power(m,i+1))-pi)])
            dist = np.array(dist)
            result[n,:,:] = dist
        rep_list[rep,:,:] = result.mean(axis=0)
    dist= rep_list.mean(axis=0)
    return dist

def plot_3d(dist):
    matplotlib.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(8, 7))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(dist[:,0],dist[:,1], dist[:,2], linewidth=0.2)
    ax.view_init(30, 30)
    ax.set_yticks(np.arange(5))
    ax.set_xlabel("Beta",labelpad=10)
    ax.set_ylabel("Iteration",labelpad=10)
    ax.set_zlabel("Error",labelpad=10)
    plt.tight_layout()
    plt.savefig("./fig/Markov_mixing.png")
    
if __name__ == '__main__':
    dist = mixing()
    plot_3d(dist)
