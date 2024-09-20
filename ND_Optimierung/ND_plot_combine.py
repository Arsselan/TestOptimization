import matplotlib.pyplot as plt
from ND_gradient_free_minize import doParticleSwarm, objective_valueP
from ND_scipy_minimize import result, objective_value
from ND_in_house_minimize import doGradientDescent

# Plot für ParticleSwarm
doParticleSwarm()
iter = []
new_range = len(objective_valueP)
for i in range(new_range):
    iter.append(i)

plt.plot(iter,objective_valueP, marker='o', label='Particle Swarm Optimization', linestyle='-', color='b')

#Plot für ScipyMinimize
iter = []
new_range = len(objective_value)
for i in range(new_range):
    iter.append(i)

print(f"Optimal solution: {result.x}")
print(f"Objective function value at optimal solution: {result.fun}")
print(f"Optimization success: {result.success}")
print(f"Message: {result.message}")

plt.plot(iter,objective_value,  marker='x', label='Scipy Optimization', linestyle='--', color='r')

#Plot für GradientDescent
iAll, fAll = doGradientDescent()

plt.plot(iAll, fAll, marker='s', label='Gradient Descent Optimization', linestyle=':', color='g')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Comparison of Optimization Algorithms')
plt.legend()
plt.grid(True)
plt.show()
