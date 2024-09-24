import numpy as np
import scipy

def objectiveFunction( x ):
    return ( x - 3 )**2 + 1

#def objectiveFunctionDerivative( x ):
 #   return 2 * x - 6

objective_valueScipy = []
def objective_function_scipy(u):
    objective_valueScipy.append(objectiveFunction(u[0]))
    return objectiveFunction(u[0])

def doScipyOptimize():
    global objective_valueScipy
    objective_valueScipy = []

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    return scipy.optimize.minimize(objective_function_scipy, np.array([50]), method='L-BFGS-B')
#es jabb ab der Methode liegen, dass er Kantig aussieht, man kann andere Methoden verwendet L-BFGS-B oder Powell

result = doScipyOptimize()

iterScipy = []
new_range = len(objective_valueScipy)
for i in range (new_range):
    iterScipy.append(i)

#erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = "1D_Scipy_results_start0.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")  # Kopfzeile
        for i in range(len(iterScipy)):
            file.write(f"{iterScipy[i]}\t{objective_valueScipy[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")