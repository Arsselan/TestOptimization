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
                    "u": np.arange(-50, 50, 0.1)
                    }
    opt = ParticleSwarmOptimizer(search_space, population=5)
    opt.search(objective_function_gfo, n_iter=500)

doParticleSwarm()
iter = []
new_range = len(objective_valueP)
for i in range(new_range):
    iter.append(i)

#erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = "ParticleSwarm_results.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")  # Kopfzeile
        for i in range(len(iter)):
            file.write(f"{iter[i]}\t{objective_valueP[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")