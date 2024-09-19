import numpy as np
import matplotlib.pyplot as plt
from gradient_free_optimizers import ParticleSwarmOptimizer


def objectiveFunction( x ):
    return ( x - 3 )**2 + 1

objective_valueP = []


def objective_function_gfo(para):
    u = para["u"]
    value = objectiveFunction(u)
    objective_valueP.append(value)
    return -value


def doParticleSwarm():
    
    search_space = {
                    "u": np.arange(-20, 20, 0.1)
                    }
    opt = ParticleSwarmOptimizer(search_space, population=5)
    opt.search(objective_function_gfo, n_iter=500)

