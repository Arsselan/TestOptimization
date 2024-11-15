import numpy as np

# Ziel-Funktion
def objectiveFunction(x):
    return (x - 3)**2 + 1

# Analytische Ableitung der Funktion (Gradient)
def objectiveFunctionDerivative(x):
    return 2 * (x - 3)

# Gradientenabstieg mit fester Schrittweite
def gradientDescent(function, derivative, initial):
    tol = 1e-5  # Konvergenztoleranz für den Funktionswert und die Änderung
    x = initial[0]
    iAll = []
    fAll = []
    alpha = 0.1  # Lernrate (Schrittweite), die ggf. angepasst werden kann

    for i in range(100):  # Maximal 100 Iterationen
        f = function(x)
        grad = derivative(x)

        # Berechne den Gradientenabstiegsschritt
        dx = -alpha * grad  # Update des Parameters x

        # Ausgabe für Debugging
        print(f"Iteration {i}: x = {x:e}, f(x) = {f:e}, grad = {grad:e}, dx = {dx:e}")
        
        iAll.append(i)
        fAll.append(f)
        
        x += dx  # Update des Parameters x

        # Überprüfung auf Konvergenz: Wenn die Änderung in x oder der Funktionswert klein genug ist
        if abs(dx) < tol and abs(f) < tol:
            print(f"Converged after {i} iterations. Solution is x = {x:e}, f(x) = {f:e}.")
            break

    return iAll, fAll

def doGradientDescent():
    initial = np.array([1000.0])  # Startwert für den Gradientenabstieg
    iAll, fAll = gradientDescent(objectiveFunction, objectiveFunctionDerivative, initial)
    return iAll, fAll, initial[0]

iAll, fAll, initial_value = doGradientDescent()

# Erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = f"1D_Gradient_start{initial_value:.0f}_fixed_alpha.txt"
    
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")  # Kopfzeile
        for i in range(len(iAll)):
            file.write(f"{iAll[i]}\t{fAll[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
