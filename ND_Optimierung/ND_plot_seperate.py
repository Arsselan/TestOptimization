import matplotlib.pyplot as plt
from ND_gradient_free_minize import  objective_valueParticle, iterParticle

plt.plot(iterParticle,objective_valueParticle)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Particle Swarm Optimization')
plt.show()

from ND_scipy_minimize import result, objective_valueScipy, iterScipy

print(f"Optimal solution: {result.x}")
print(f"Objective function value at optimal solution: {result.fun}")
print(f"Optimization success: {result.success}")
print(f"Message: {result.message}")

plt.plot(iterScipy,objective_valueScipy)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Scipy Optimization')
plt.show()

from ND_in_house_minimize import iAll, fAll

plt.plot(iAll, fAll, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Gradient Descent Optimization')
plt.grid(True)
plt.show()
