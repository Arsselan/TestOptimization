import numpy as np
from gradient_free_optimizers import ParticleSwarmOptimizer

# Beispiel der Zielfunktion
def objectiveFunction(x):
    return (x - 3) ** 2 + 1

# Liste für die gespeicherten Werte
objective_valueParticle = []

# Diese Funktion wird von PSO verwendet
def objective_function_gfo(para):
    u = para["u"]
    value = objectiveFunction(u)
    objective_valueParticle.append(value)
    return -value

# Parameter für den Suchraum
start = -20
end = 100
step = 1 
pop = 500  # Erhöhe die Population

def doParticleSwarm():
    search_space = {
        "u": np.arange(start, end, step)
    }
    opt = ParticleSwarmOptimizer(search_space, population=pop)  # Population größer als 1
    print("Starte die Optimierung...")  # Debugging-Ausgabe
    opt.search(objective_function_gfo, n_iter=100)  # Stelle sicher, dass 100 Iterationen durchgeführt werden
    print("Optimierung abgeschlossen!")  # Debugging-Ausgabe

doParticleSwarm()

# Überprüfe, ob Daten in der Liste gespeichert wurden
if len(objective_valueParticle) == 0:
    print("Fehler: Es wurden keine Werte in objective_valueParticle gespeichert.")
else:
    print(f"Anzahl der gespeicherten Werte: {len(objective_valueParticle)}")  # Debug-Ausgabe

# Iterationen und Ergebnisse speichern
iterParticle = list(range(len(objective_valueParticle)))

# Erstelle eine Textdatei mit den Ergebnissen
if __name__ == "__main__":
    filename = f"ParticleSwarm_start={start}_end={end}_step={step}_pop={pop}.txt"  # Korrekte Variablen im Dateinamen
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")  # Kopfzeile
        for i in range(len(iterParticle)):
            file.write(f"{iterParticle[i]}\t{objective_valueParticle[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
