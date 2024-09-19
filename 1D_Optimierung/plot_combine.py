import numpy as np
import matplotlib.pyplot as plt
from scipy_minimize import doScipyOptimize, objective_value, result
from gradient_free_minimize import doParticleSwarm, objective_valueP
from in_house_minimize import doGradientDescent

def plot_optimization_results():
   
    result = doScipyOptimize()
    doParticleSwarm()
    iAll_gd, fAll_gd = doGradientDescent()

    plt.figure(figsize=(12, 8))

    # Plot für Scipy 
    plt.plot(objective_value, marker='o', label='Scipy Optimization', linestyle='-', color='b')

    # Plot für ParticleSwarm 
    pso_iter = list(range(len(objective_valueP)))
    plt.plot(pso_iter, objective_valueP, marker='x', label='Particle Swarm Optimization', linestyle='--', color='r')

    # Plot für GradientDescent 
    plt.plot(iAll_gd, fAll_gd, marker='s', label='Gradient Descent Optimization', linestyle=':', color='g')

    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Comparison of Optimization Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()
    
plot_optimization_results()
