import matplotlib.pyplot as plt
from scipy_minimize import objective_valueScipy, result, iterScipy

print(f"Optimal solution: {result.x}")
print(f"Objective function value at optimal solution: {result.fun}")
print(f"Optimization success: {result.success}")
print(f"Message: {result.message}")

plt.plot(iterScipy, objective_valueScipy, marker = 'o')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Scipy Optimization')
plt.show()

from gradient_free_minimize import iterParticle, objective_valueParticle
plt.plot(iterParticle,objective_valueParticle)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Particle Swarm Optimization')
plt.show()

from in_house_minimize import iAll, fAll
plt.plot(iAll,fAll)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Gradient Descent Optimization')
plt.show()