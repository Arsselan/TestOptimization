import numpy as np

# Ziel-Funktion für mehrere Variablen
def objectiveFunction(u, v):
    return (u - 2)**2 + (v + 3)**2 + np.sin(u**2 + v**2)

# Partielle Ableitungen der Ziel-Funktion
def objectiveFunctionDerivative(u, v):
    df_du = 2 * (u - 2) + 2 * u * np.cos(u**2 + v**2)  # Partielle Ableitung nach u
    df_dv = 2 * (v + 3) + 2 * v * np.cos(u**2 + v**2)  # Partielle Ableitung nach v
    return np.array([df_du, df_dv])

# Gradientenabstieg mit fester Schrittweite
def gradientDescent(function, derivative, initial):
    tol = 1e-5  # Konvergenztoleranz für den Funktionswert und die Änderung
    u, v = initial  # Entpacken des Initialwerts
    iAll = []
    fAll = []
    alpha = 0.1  # Lernrate (Schrittweite), die ggf. angepasst werden kann

    for i in range(10000):  # Maximal 100 Iterationen
        f = function(u, v)
        grad = derivative(u, v)

        # Berechne den Gradientenabstiegsschritt
        du, dv = -alpha * grad  # Update des Parameters u und v

        # Ausgabe für Debugging
        print(f"Iteration {i}: u = {u:e}, v = {v:e}, f(u, v) = {f:e}, grad = {grad}, du = {du:e}, dv = {dv:e}")
        
        iAll.append(i)
        fAll.append(f)
        
        u += du  # Update des Parameters u
        v += dv  # Update des Parameters v

        # Überprüfung auf Konvergenz: Wenn die Änderung in u, v oder der Funktionswert klein genug ist
        if np.linalg.norm([du, dv]) < tol and abs(f) < tol:
            print(f"Converged after {i} iterations. Solution is u = {u:e}, v = {v:e}, f(u, v) = {f:e}.")
            break

    return iAll, fAll

def doGradientDescent():
    initial = np.array([50.0, 50.0])  # Startwerte für den Gradientenabstieg
    iAll, fAll = gradientDescent(objectiveFunction, objectiveFunctionDerivative, initial)
    return iAll, fAll, initial

iAll, fAll, initial_value = doGradientDescent()

# Erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = f"2D_Gradient_start{initial_value[0]:.0f}_{initial_value[1]:.0f}_fixed_alpha.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")
        for i in range(len(iAll)):
            file.write(f"{iAll[i]}\t{fAll[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
