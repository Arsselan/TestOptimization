import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize



def objectiveFunction( x ):
    return ( x - 3 )**2 + 1+ np.sin(x)


#def objectiveFunctionDerivative( x ):
 #   return 2 * x - 6


objective_value = []
def objective_function_scipy(u):
    objective_value.append(objectiveFunction(u[0]))
    return objectiveFunction(u[0])


def doScipyOptimize():
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    return scipy.optimize.minimize(objective_function_scipy, np.array([0]), method='L-BFGS-B')
#es jabb ab der Methode liegen, dass er Kantig aussieht, man kann andere Methoden verwendet L-BFGS-B oder Powell
result = doScipyOptimize()

print(f"Optimal solution: {result.x}")
print(f"Objective function value at optimal solution: {result.fun}")
print(f"Optimization success: {result.success}")
print(f"Message: {result.message}")


plt.plot(objective_value, marker = 'o')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.show()


