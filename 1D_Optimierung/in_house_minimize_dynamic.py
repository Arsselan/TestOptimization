import numpy as np

# Die zu minimierende Funktion
def objectiveFunction(x):
    return (x - 3)**2 + 1

# Die Ableitung der Funktion
def objectiveFunctionDerivative(x):
    return 2 * x - 6

# Numerische Berechnung des Gradienten (Tangent)
def numericTangent(function, x, eps=1e-6):
    f = function(x)
    n = x.shape[0]
    t = np.zeros_like(x)
    for i in range(n):
        iEps = eps * abs(x[i])
        xp = x.copy()  # Kopiere den Punkt, um Änderungen vorzunehmen
        xp[i] += iEps
        deltaF = function(xp) - f
        if abs(deltaF) < 1e-10:
            print("Warning! Delta too small.")
        t[i] = deltaF / iEps
    return f.item(), t  # Extrahiere den Skalarwert für f

# Armijo-Regel zur Bestimmung der Schrittgröße
def armijo_rule(f, grad_f, x, alpha, c=0.5):
    while f(x - alpha * grad_f(x)) > f(x) - c * alpha * np.linalg.norm(grad_f(x))**2:
        alpha *= 0.5  # Halbiert die Schrittgröße, wenn die Bedingung nicht erfüllt ist
    return alpha

# Gradientenabstieg mit Armijo-Regel und numerischer Ableitung
def gradientDescent(function, initial):
    tol = 1e-5
    x = initial[0]
    iAll = []
    fAll = []
    alpha = 0.5  # Initiale Schrittgröße
    for i in range(100):
        # Berechne die numerische Ableitung an der aktuellen Stelle
        f, t = numericTangent(function, np.array([x]), 1e-5)  # Funktionswert und Gradient berechnen
        # Berechne Schrittgröße mit der Armijo-Regel
        alpha = armijo_rule(function, lambda x: numericTangent(function, np.array([x]), 1e-7)[1], np.array([x]), alpha)
        
        # Berechne den Abstiegsschritt
        dx = -alpha * t
        
        # Extrahiere skalare Werte für die Ausgabe
        print(f"{i}: x = {x:e}, f(x) = {f:e}, t = {np.linalg.norm(t):e}, dx = {np.linalg.norm(dx):e}, alpha = {alpha:e}")
        
        iAll.append(i)
        fAll.append(f)
        
        # Update der Position x
        x += dx[0]  # Update x
        
        # Abbruchbedingung, wenn der Schritt klein genug ist
        if np.linalg.norm(dx) < tol:  
            print(f"Converged after {i} iterations. Solution is x = {x:e}.")
            break
    
    return iAll, fAll

# Hauptfunktion, die den Gradientenabstieg startet
def doGradientDescent():
    initial = np.array([100.0])  # Startwert
    iAll, fAll = gradientDescent(objectiveFunction, initial)
    return iAll, fAll

# Ergebnisse speichern
iAll, fAll = doGradientDescent()

# Erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = "1D_Gradient_results_armijo.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")
        for i in range(len(iAll)):
            file.write(f"{iAll[i]}\t{fAll[i]:.6f}\n")

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
