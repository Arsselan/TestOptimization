import matplotlib.pyplot as plt
from ND_gradient_free_minize import doParticleSwarm, objective_valueP
'''
doParticleSwarm()
iter = []
new_range = len(objective_valueP)
for i in range(new_range):
    iter.append(i)
'''
plt.plot(iter,objective_valueP)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Particle Swarm Optimization')
plt.show()

from ND_scipy_minimize import result, objective_value

iter = []
new_range = len(objective_value)
for i in range(new_range):
    iter.append(i)

print(f"Optimal solution: {result.x}")
print(f"Objective function value at optimal solution: {result.fun}")
print(f"Optimization success: {result.success}")
print(f"Message: {result.message}")

plt.plot(iter,objective_value)
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
