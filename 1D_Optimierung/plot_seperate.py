import matplotlib.pyplot as plt

from scipy_minimize import doScipyOptimize, objective_value, result

print(f"Optimal solution: {result.x}")
print(f"Objective function value at optimal solution: {result.fun}")
print(f"Optimization success: {result.success}")
print(f"Message: {result.message}")

plt.plot(objective_value, marker = 'o')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.show()


from gradient_free_minimize import doParticleSwarm, objective_valueP

doParticleSwarm()
iter = []
new_range = len(objective_valueP)
for i in range(new_range):
    iter.append(i)

plt.plot(iter,objective_valueP)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.show()


from in_house_minimize import  doGradientDescent
iAll,fAll = doGradientDescent()

plt.plot(iAll,fAll)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.show()