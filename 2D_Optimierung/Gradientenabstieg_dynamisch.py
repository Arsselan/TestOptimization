import numpy as np

# Ziel-Funktion
def objectiveFunction(x):
    u, v = x
    return (u - 2)**2 + (v + 3)**2 + np.sin(u**2 + v**2)

# Ableitung der Ziel-Funktion
def objectiveFunctionDerivative(x):
    u, v = x
    du = 2 * (u - 2) + 2 * u * np.cos(u**2 + v**2)  # Ableitung nach u
    dv = 2 * (v + 3) + 2 * v * np.cos(u**2 + v**2)  # Ableitung nach v
    return np.array([du, dv])

# Armijo-Regel zur Bestimmung der Schrittgröße
def armijo_rule(f, grad_f, x, alpha, c=0.5):
    grad = grad_f(x)  # Berechne den Gradienten an der aktuellen Stelle
    # Überprüfe die Armijo-Bedingung und passe alpha an
    while f(x - alpha * grad) > f(x) - c * alpha * np.linalg.norm(grad)**2:
        alpha *= 0.5  # Halbiert die Schrittgröße, wenn die Bedingung nicht erfüllt ist
    return alpha

def gradientDescent(function, derivative, initial):
    tol = 1e-5  # Konvergenztoleranz für den Funktionswert und die Änderung
    x = initial  # Startwert (für beide Variablen u und v)
    iAll = []
    fAll = []
    alpha = 0.1  # Initiale Schrittgröße

    for i in range(100):  # Maximal 100 Iterationen
        f = function(x)
        grad = derivative(x)

        # Berechne Schrittgröße mit der Armijo-Regel
        alpha = armijo_rule(function, derivative, x, alpha)
        
        # Berechne den Abstiegsschritt
        dx = -alpha * grad
        
        # Extrahiere skalare Werte für die Ausgabe
        print(f"{i}: x = {x}, f(x) = {f}, grad = {grad}, dx = {dx}, alpha = {alpha}")
        
        iAll.append(i)
        fAll.append(f)
        
        # Update der Position x
        x += dx  # Update x
        
        # Abbruchbedingung, wenn der Schritt klein genug ist
        if np.linalg.norm(dx) < tol:  
            print(f"Converged after {i} iterations. Solution is x = {x}, f(x) = {f}.")
            break
    
    return iAll, fAll

def doGradientDescent():
    initial = np.array([1000.0, 1000.0])  # Startwert für den Gradientenabstieg (für u und v)
    iAll, fAll = gradientDescent(objectiveFunction, objectiveFunctionDerivative, initial)
    return iAll, fAll, initial

# Ergebnisse speichern
iAll, fAll, initial_value = doGradientDescent()

# Erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = f"2D_Gradient_start{initial_value[0]:.0f}_{initial_value[1]:.0f}_dynamic_alpha.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")
        for i in range(len(iAll)):
            file.write(f"{iAll[i]}\t{fAll[i]:.6f}\n")

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
