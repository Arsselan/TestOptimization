import numpy as np
import matplotlib.pyplot as plt
from gradient_free_optimizers import ParticleSwarmOptimizer


def objectiveFunction( x , y):
    return ( x - 3 )**2 + 1 + np.sin(y) +x*y

objective_value = []


def objective_function_gfo(para):
    u = para["u"]
    u2 = para["u2"]
    value = objectiveFunction(u,u2)
    objective_value.append(value)
    return -value


def doParticleSwarm():
    
    search_space = {
                    "u": np.arange(-20, 20, 0.1),
                    "u2": np.arange(-20, 20, 0.1)
                    }
    opt = ParticleSwarmOptimizer(search_space, population=5)
    opt.search(objective_function_gfo, n_iter=500)

doParticleSwarm()
iter = []
new_range = len(objective_value)
for i in range(new_range):
    iter.append(i)

plt.plot(iter,objective_value)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.show()


