import numpy as np
from gradient_free_optimizers import ParticleSwarmOptimizer


def objectiveFunction( x , y):
    return (x-2)**2 +(y+3)**2 +np.sin(x**2 +y**2)
    #Globales Minimum bei f(x,y) = (2,-3)
    #return np.sin(x**2+y**2)+0.5*(x**2+y**2)-0.5
    # Globales Minimum bei f(x,y) = (0,0)

objective_valueParticle = []


def objective_function_gfo(para):
    u = para["u"]
    u2 = para["u2"]
    value = objectiveFunction(u,u2)
    objective_valueParticle.append(value)
    return -value


def doParticleSwarm():
    
    search_space = {
                    "u": np.arange(-50, 50, 0.1),
                    "u2": np.arange(-50, 50, 0.1)
                    }
    opt = ParticleSwarmOptimizer(search_space, population=5)
    opt.search(objective_function_gfo, n_iter=500)


doParticleSwarm()
iterParticle = []
new_range = len(objective_valueParticle)
for i in range(new_range):
    iterParticle.append(i)

#erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = "ND_ParticleSwarm_results.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")  # Kopfzeile
        for i in range(len(iterParticle)):
            file.write(f"{iterParticle[i]}\t{objective_valueParticle[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
