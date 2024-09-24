import matplotlib.pyplot as plt
from ND_gradient_free_minize import iterParticle, objective_valueParticle
from ND_scipy_minimize import result, objective_valueScipy, iterScipy
from ND_in_house_minimize import iAll, fAll

# Plot für ParticleSwarm

plt.plot(iterParticle,objective_valueParticle, marker='o', label='Particle Swarm Optimization', linestyle='-', color='b')

#Plot für ScipyMinimize

print(f"Optimal solution: {result.x}")
print(f"Objective function value at optimal solution: {result.fun}")
print(f"Optimization success: {result.success}")
print(f"Message: {result.message}")

plt.plot(iterScipy,objective_valueScipy,  marker='x', label='Scipy Optimization', linestyle='--', color='r')

#Plot für GradientDescent

plt.plot(iAll, fAll, marker='s', label='Gradient Descent Optimization', linestyle=':', color='g')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Comparison of Optimization Algorithms')
plt.legend()
plt.grid(True)
plt.show()
