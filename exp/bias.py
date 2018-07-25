import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#### TO FIX THE GRAPH NOT EXACTLY THE SAME AS THE PAPER


def get_bias():
    data = np.zeros((50,100,4))
    size = 10
    T = 5
    for p in range(100):
        dist = []
        dist_mod = []
        for i in range(50):
            matrix = np.random.rand(size,size)
            matrix = matrix/matrix.sum(axis=1)[:,None]
            matrix_t = (matrix.T/matrix.T.sum(axis=1)[:,None])
            matrix_t = matrix.dot(matrix_t)
            matrix_t_m = ((p+1)/100.0)*matrix_t + (1-((p+1)/100.0))*matrix

            reward = np.random.rand(size)
            reward_mod = copy.copy(reward)
            for u in range(3):
                for n in range(size):
                    ind = np.argpartition(matrix[:,n], -(u+2))[-(u+2):]
                    for s in range(u+2):
                        reward_mod[ind[s]] += (reward_mod[n] - reward_mod[ind[s]])/2

                sol_mod = np.linalg.solve(np.identity(size)-0.99*matrix,reward_mod)
                sol_2_mod = np.linalg.solve(np.identity(size)-0.99*matrix_t_m,reward_mod)
                data[i,p,u+1] = np.abs(sol_mod - sol_2_mod).mean()


            sol = np.linalg.solve(np.identity(size)-0.99*matrix,reward)
            sol_2 = np.linalg.solve(np.identity(size)-0.99*matrix_t_m,reward)
            data[i,p,0] = np.abs(sol - sol_2).mean()
    return data

def plot(data):        
    ax = sns.tsplot(data,condition=["N = 0","N = 2","N = 3","N = 4"])
    ax.set_xlabel(r'values of $\beta$')
    n = len(ax.xaxis.get_ticklabels())           
    ax.set_xticklabels(np.linspace(0, 1, n))
    plt.savefig("./fig/bias.png")
if __name__ == '__main__':
    data = get_bias()
    plot(data)