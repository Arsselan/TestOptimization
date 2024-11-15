import numpy as np

# Ziel-Funktion für mehrere Variablen (u, v)
def objectiveFunction(u, v):
    return (u - 2)**2 + (v + 3)**2 + np.sin(u**2 + v**2)

# Partielle Ableitungen der Ziel-Funktion (Gradient)
def objectiveFunctionDerivative(u, v):
    df_du = 2 * (u - 2) + 2 * u * np.cos(u**2 + v**2)  # Partielle Ableitung nach u
    df_dv = 2 * (v + 3) + 2 * v * np.cos(u**2 + v**2)  # Partielle Ableitung nach v
    return np.array([df_du, df_dv])

# Armijo-Regel zur Bestimmung der Schrittgröße
def armijo_rule(f, grad_f, x, alpha, c=0.5, beta=0.1):
    grad = grad_f(*x)  # Berechne den Gradienten an der aktuellen Stelle (für mehrere Variablen)
    # Überprüfe die Armijo-Bedingung und passe alpha an
    while f(*(x - alpha * grad)) > f(*x) - c * alpha * np.linalg.norm(grad)**2:
        alpha *= beta  # Reduziere die Schrittgröße, wenn die Bedingung nicht erfüllt ist
    return alpha

# Gradientenabstieg mit dynamischer Schrittgröße
def gradientDescent(function, derivative, initial):
    tol = 1e-5  # Konvergenztoleranz für den Funktionswert und die Änderung
    u, v = initial  # Entpacken der Initialwerte
    iAll = []
    fAll = []
    alpha = 0.1  # Initiale Schrittgröße
    c = 0.5  # Armijo-Konstanten
    beta = 0.5  # Skalierungsfaktor für alpha bei Nicht-Erfüllung der Armijo-Bedingung

    for i in range(10000):  # Maximal 100 Iterationen
        x = np.array([u, v])
        f = function(u, v)
        grad = derivative(u, v)

        # Berechne Schrittgröße mit der Armijo-Regel
        alpha = armijo_rule(function, derivative, x, alpha, c, beta)
        
        # Berechne den Abstiegsschritt
        du, dv = -alpha * grad  # Update der Parameter u und v
        
        # Ausgabe für Debugging
        print(f"Iteration {i}: u = {u:e}, v = {v:e}, f(u, v) = {f:e}, grad = {grad}, du = {du:e}, dv = {dv:e}, alpha = {alpha:e}")
        
        iAll.append(i)
        fAll.append(f)
        
        # Update der Position (u, v)
        u += du
        v += dv
        
        # Abbruchbedingung, wenn der Schritt klein genug ist
        if np.linalg.norm([du, dv]) < tol and abs(f) < tol:
            print(f"Converged after {i} iterations. Solution is u = {u:e}, v = {v:e}, f(u, v) = {f:e}.")
            break
    
    return iAll, fAll

def doGradientDescent():
    initial = np.array([50.0, 50.0])  # Startwerte für den Gradientenabstieg
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
            file.write(f"{iAll[i]}\t{fAll[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
