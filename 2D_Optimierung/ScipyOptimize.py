import numpy as np
import scipy.optimize

# 2D Zielfunktion
def objectiveFunction(x):
    u, v = x
    return (u - 2)**2 + (v + 3)**2 + np.sin(u**2 + v**2)

# Ableitungen der Zielfunktion (Gradienten)
def objectiveFunctionDerivative(x):
    u, v = x
    du = 2 * (u - 2) + 2 * u * np.cos(u**2 + v**2)  # Partielle Ableitung nach u
    dv = 2 * (v + 3) + 2 * v * np.cos(u**2 + v**2)  # Partielle Ableitung nach v
    return np.array([du, dv])

# Liste für die gespeicherten Werte
objective_valueScipy = []

start_value = None  # Globale Variable für den Startwert

# Diese Funktion wird von scipy.optimize.minimize verwendet
def objective_function_scipy(u):
    # Speichern des Funktionswertes für jede Iteration
    objective_valueScipy.append(objectiveFunction(u))
    grad = objectiveFunctionDerivative(u)  # Berechne Gradienten
    return objectiveFunction(u), grad  # Funktion gibt Wert und Gradienten zurück

def doScipyOptimize():
    global objective_valueScipy, start_value  # Zugriff auf globale Variablen
    objective_valueScipy = []  # Reset der gespeicherten Werte

    # Startwert definieren und speichern (für u und v)
    start_value = [10000, 10000]  # Startwerte für u und v
    initial = np.array(start_value)

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
    filename = f"2D_Scipy_start={start_value[0]}_{start_value[1]}.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")  # Kopfzeile
        for i in range(len(iterScipy)):
            file.write(f"{iterScipy[i]}\t{objective_valueScipy[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
