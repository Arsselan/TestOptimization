import numpy as np
import scipy
from scipy import optimize



def objectiveFunction( x ):
    return ( x - 3 )**2 + 1


#def objectiveFunctionDerivative( x ):
 #   return 2 * x - 6


objective_value = []
def objective_function_scipy(u):
    objective_value.append(objectiveFunction(u[0]))
    return objectiveFunction(u[0])


def doScipyOptimize():
    global objective_value
    objective_value = []

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    return scipy.optimize.minimize(objective_function_scipy, np.array([0]), method='L-BFGS-B')
#es jabb ab der Methode liegen, dass er Kantig aussieht, man kann andere Methoden verwendet L-BFGS-B oder Powell

result = doScipyOptimize()


