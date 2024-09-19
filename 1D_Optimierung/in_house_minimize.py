import numpy as np
import matplotlib.pyplot as plt


def objectiveFunction( x ):
    return ( x - 3 )**2 + 1


def objectiveFunctionDerivative( x ):
    return 2 * x - 6


def plot(x, y):
    plt.plot(x, y, "-o")
    plt.xlabel('Iteration')
    plt.ylabel('Obejective Function Value')
    plt.title('Gradient Descent')
    plt.show()


def numericTangent( function, x, eps=1e-6 ):
    f = function(x)
    n = x.shape[0]
    t = np.zeros_like(x)
    for i in range(n):
        iEps = eps * abs(x[i])
        xp = x.copy() #copy fürs copieren verwenden
        xp[i] += iEps
        deltaF = function( xp ) - f
        if abs(deltaF) < 1e-10:
            print("Waring! Delta too small.")
        t[i] = deltaF / iEps
    return f, t


def gradientDescent( function, initial ):
    tol = 1e-5
    x = initial[0]
    iAll = []
    fAll = []
    for i in range(100):
        alpha = 0.1
        #f, t = numericTangent( function, np.ndarray(x), 1e-7 )
        f = objectiveFunction(x)
        t = objectiveFunctionDerivative(x)
        dx = - alpha * t
        print(f"{i}: x = {x:e}, f(x) = {f:e}, t = {t:e}, dx = {dx:e}")
        iAll.append(i)
        fAll.append(f)
        x += dx
        if abs(dx) < tol:
            print(f"Converged after {i} iterations. Solution is x = {x:e}.")
            break
    return iAll, fAll


def doGradientDescent():
    initial = np.array([50.0])
    iAll, fAll = gradientDescent( objectiveFunction, initial )
    plot(iAll, fAll)

doGradientDescent()





