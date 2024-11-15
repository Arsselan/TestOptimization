import numpy as np

# Ziel-Funktion (2D)
def objectiveFunction(x):
    u, v = x[0], x[1]
    return (u - 2)**2 + (v + 3)**2 + np.sin(u**2 + v**2)

# Analytische Ableitung der Funktion (Gradient) - 2D
def objectiveFunctionDerivative(x):
    u, v = x[0], x[1]
    du = 2 * (u - 2) + 2 * u * np.cos(u**2 + v**2)  # Ableitung bzgl. u
    dv = 2 * (v + 3) + 2 * v * np.cos(u**2 + v**2)  # Ableitung bzgl. v
    return np.array([du, dv])

# Gradientenabstieg mit fester Schrittweite
def gradientDescent(function, derivative, initial):
    tol = 1e-5  # Konvergenztoleranz für den Funktionswert und die Änderung
    x = initial
    iAll = []
    fAll = []
    alpha = 0.1  # Lernrate (Schrittweite), die ggf. angepasst werden kann

    for i in range(100):  # Maximal 100 Iterationen
        f = function(x)
        grad = derivative(x)

        # Berechne den Gradientenabstiegsschritt
        dx = -alpha * grad  # Update des Parameters x

        # Ausgabe für Debugging
        print(f"Iteration {i}: x = {x}, f(x) = {f:e}, grad = {grad}, dx = {dx}")
        
        iAll.append(i)
        fAll.append(f)
        
        x += dx  # Update des Parameters x

        # Überprüfung auf Konvergenz: Wenn die Änderung in x oder der Funktionswert klein genug ist
        if np.linalg.norm(dx) < tol and abs(f) < tol:
            print(f"Converged after {i} iterations. Solution is x = {x}, f(x) = {f:e}.")
            break

    return iAll, fAll

def doGradientDescent():
    initial = np.array([1000.0, 1000.0])  # Startwert für den Gradientenabstieg (2D)
    iAll, fAll = gradientDescent(objectiveFunction, objectiveFunctionDerivative, initial)
    return iAll, fAll, initial

iAll, fAll, initial_value = doGradientDescent()

# Erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = f"2D_Gradient_start{initial_value[0]:.0f}_{initial_value[1]:.0f}_fixed_alpha.txt"
    
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")  # Kopfzeile
        for i in range(len(iAll)):
            file.write(f"{iAll[i]}\t{fAll[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
