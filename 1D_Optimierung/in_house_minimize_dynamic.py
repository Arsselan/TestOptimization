import numpy as np

def objectiveFunction(x):
    return (x - 3)**2 + 1

def numericTangent(function, x, eps=1e-6):
    f = function(x)
    t = np.zeros_like(x)
    for i in range(x.shape[0]):
        iEps = eps * abs(x[i])
        xp = x.copy()
        xp[i] += iEps
        deltaF = function(xp) - f
        if abs(deltaF) < 1e-10:
            print("Warning! Delta too small.")
        t[i] = (deltaF / iEps).item()  # Verwenden Sie .item() um den Skalarwert zu extrahieren
    return f, t

def gradientDescent(function, initial):
    tol = 1e-5
    x = initial[0]
    iAll = []
    fAll = []
    
    gamma = 0.1  # Bruchteil für die Armijo-Bedingung
    alpha = 1.0  # Anfängliche Schrittweite
    alpha_decay = 0.5  # Faktor zur Verkürzung der Schrittweite
    
    for i in range(100):
        f, t = numericTangent(function, np.array([x]), 1e-7)
        t = t[0]  # Extrahiere den Skalarwert aus dem Array
        d = -t  # Richtung des steilsten Abstiegs
        
        # Armijo-Bedingung
        while function(x + alpha * d) > f + gamma * alpha * t * d:
            alpha *= alpha_decay
        
        dx = alpha * d
        #print(f"{i}: x = {x:e}, f(x) = {f:e}, t = {t:e}, dx = {dx:e}, alpha = {alpha:e}")
        iAll.append(i)
        fAll.append(f)
        x += dx
        
        if abs(dx) < tol:
            print(f"Konvergiert nach {i} Iterationen. Lösung ist x = {x:e}.")
            break
        
        alpha = 1.0  # Setze alpha für die nächste Iteration zurück
    
    return iAll, fAll

def doGradientDescent():
    initial = np.array([50.0])
    iAll, fAll = gradientDescent(objectiveFunction, initial)
    return iAll, fAll

iAll, fAll = doGradientDescent()

# Erstellen einer Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = "1D_Gradient_results_start0_armijo.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")
        for i in range(len(iAll)):
            file.write(f"{iAll[i]}\t{fAll[i].item():.6f}\n")

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")