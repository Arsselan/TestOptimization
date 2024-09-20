import numpy as np
import matplotlib.pyplot as plt

def objectiveFunction(x):
    u, v = x
    return (u - 2)**2 + (v + 3)**2 + np.sin(u**2 + v**2)

def objectiveFunctionDerivative(x):
    u, v = x
    df_du = 2 * (u - 2) + 2 * u * np.cos(u**2 + v**2)
    df_dv = 2 * (v + 3) + 2 * v * np.cos(u**2 + v**2)
    return np.array([df_du, df_dv])

def gradientDescent(function, initial):
    tol = 1e-5
    x = initial
    iAll = []
    fAll = []
    for i in range(500):
        alpha = 0.1
        f = function(x)
        t = objectiveFunctionDerivative(x)
        dx = - alpha * t
        print(f"Iteration {i}: x = {x}, f(x) = {f:e}, gradient = {t}, dx = {dx}")
        iAll.append(i)
        fAll.append(f)
        x += dx
        if np.linalg.norm(dx) < tol:
            print(f"Converged after {i} iterations. Solution is x = {x}.")
            break
    # Ensure that the final iteration is included
    if len(iAll) == 0 or (iAll[-1] != i):
        iAll.append(i)
        fAll.append(f)
    return iAll, fAll

def doGradientDescent():
    initial = np.array([50.0, 50.0])
    iAll, fAll = gradientDescent(objectiveFunction, initial)
    return iAll, fAll
'''
iAll, fAll = doGradientDescent()

# Plotting the results
plt.plot(iAll, fAll, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Gradient Descent Optimization')
plt.grid(True)
plt.show()
'''