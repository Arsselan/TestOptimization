import numpy as np
import scipy.optimize

# Zielfunktion
def objectiveFunction(x):
    return (x - 3)**2 + 1

# Ableitung der Zielfunktion (Gradient)
def objectiveFunctionDerivative(x):
    return 2 * (x - 3)  # Ableitung von (x - 3)**2 + 1

# Liste für die gespeicherten Werte
objective_valueScipy = []

start_value = None  # Globale Variable für den Startwert

# Diese Funktion wird von scipy.optimize.minimize verwendet
def objective_function_scipy(u):
    # Speichern des Funktionswertes für jede Iteration
    objective_valueScipy.append(objectiveFunction(u[0]))
    return objectiveFunction(u[0]), objectiveFunctionDerivative(u[0])  # Funktion gibt Wert und Gradienten zurück

def doScipyOptimize():
    global objective_valueScipy, start_value  # Zugriff auf globale Variablen
    objective_valueScipy = []  # Reset der gespeicherten Werte

    # Startwert definieren und speichern
    start_value = 10000
    initial = np.array([start_value])

    # Optimierung durchführen (L-BFGS-B verwendet die Gradienten)
    result = scipy.optimize.minimize(objective_function_scipy, initial, method='L-BFGS-B', jac=True)
    return result

result = doScipyOptimize()

# Iterationen und Ergebnisse speichern
iterScipy = []
new_range = len(objective_valueScipy)
for i in range(new_range):
    iterScipy.append(i)

# Erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    # Dynamischer Dateiname mit Startwert
    filename = f"1D_Scipy_start={start_value}.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")  # Kopfzeile
        for i in range(len(iterScipy)):
            file.write(f"{iterScipy[i]}\t{objective_valueScipy[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
