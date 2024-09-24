import numpy as np
import matplotlib.pyplot as plt
from scipy_minimize import  objective_valueScipy, iterScipy
from gradient_free_minimize import  objective_valueParticle
from in_house_minimize import iAll,fAll

def plot_optimization_results():

    plt.figure(figsize=(12, 8))

    # Plot für Scipy 
    plt.plot(iterScipy, objective_valueScipy, marker='o', label='Scipy Optimization', linestyle='-', color='b')

    # Plot für ParticleSwarm 
    pso_iter = list(range(len(objective_valueParticle)))
    plt.plot(pso_iter, objective_valueParticle, marker='x', label='Particle Swarm Optimization', linestyle='--', color='r')

    # Plot für GradientDescent 
    plt.plot(iAll, fAll, marker='s', label='Gradient Descent Optimization', linestyle=':', color='g')

    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Comparison of Optimization Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()
    
plot_optimization_results()
