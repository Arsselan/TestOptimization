import numpy as np
import matplotlib.pyplot as plt
from gradient_free_optimizers import ParticleSwarmOptimizer


def objectiveFunction( x , y):
    return (x-2)**2 +(y+3)**2 +np.sin(x**2 +y**2)
    #Globales Minimum bei f(x,y) = (2,-3)
    #return np.sin(x**2+y**2)+0.5*(x**2+y**2)-0.5
    # Globales Minimum bei f(x,y) = (0,0)

objective_valueP = []


def objective_function_gfo(para):
    u = para["u"]
    u2 = para["u2"]
    value = objectiveFunction(u,u2)
    objective_valueP.append(value)
    return -value


def doParticleSwarm():
    
    search_space = {
                    "u": np.arange(-20, 20, 0.1),
                    "u2": np.arange(-20, 20, 0.1)
                    }
    opt = ParticleSwarmOptimizer(search_space, population=5)
    opt.search(objective_function_gfo, n_iter=500)

'''
doParticleSwarm()
iter = []
new_range = len(objective_valueP)
for i in range(new_range):
    iter.append(i)

plt.plot(iter,objective_valueP)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.show()
'''