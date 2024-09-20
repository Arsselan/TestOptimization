import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt



def objectiveFunction( x , y):
    return (x-2)**2 +(y+3)**2 +np.sin(x**2 +y**2)

#def objectiveFunctionDerivative( x ):
 #   return 2 * x - 6d


objective_value = []
def objective_function_scipy(param):
    u,y = param
    objective_value.append(objectiveFunction(u,y))
    return objectiveFunction(u,y)


def doScipyOptimize():
    global objective_value
    objective_value = []

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    return scipy.optimize.minimize(objective_function_scipy, np.array([0,0]), method='L-BFGS-B')
#es jabb ab der Methode liegen, dass er Kantig aussieht, man kann andere Methoden verwendet L-BFGS-B oder Powell
result = doScipyOptimize()
'''
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
plt.show()
'''