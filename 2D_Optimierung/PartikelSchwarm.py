import numpy as np
from gradient_free_optimizers import ParticleSwarmOptimizer

# Beispiel der 2D-Zielfunktion
def objectiveFunction(u, v):
    return (u - 2) ** 2 + (v + 3) ** 2 + np.sin(u**2 + v**2)

# Liste für die gespeicherten Werte
objective_valueParticle = []

# Diese Funktion wird von PSO verwendet
def objective_function_gfo(para):
    u = para["u"]
    v = para["v"]
    value = objectiveFunction(u, v)  # Jetzt mit u und v
    objective_valueParticle.append(value)
    return -value

# Parameter für den Suchraum
start_u = -20
end_u = 100
step_u = 1
start_v = -20
end_v = 100
step_v = 1
pop = 500  # Population

def doParticleSwarm():
    # Der Suchraum wird jetzt für beide Variablen (u, v) definiert
    search_space = {
        "u": np.arange(start_u, end_u, step_u),
        "v": np.arange(start_v, end_v, step_v)
    }
    opt = ParticleSwarmOptimizer(search_space, population=pop)
    print("Starte die Optimierung...")  # Debugging-Ausgabe
    opt.search(objective_function_gfo, n_iter=100)  # 100 Iterationen
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
    filename = f"2D_ParticleSwarm_start_u={start_u}_end_u={end_u}_step_u={step_u}_start_v={start_v}_end_v={end_v}_step_v={step_v}_pop={pop}.txt"
    with open(filename, 'w') as file:
        file.write("Iteration\tObjective Value (J)\n")  # Kopfzeile
        for i in range(len(iterParticle)):
            file.write(f"{iterParticle[i]}\t{objective_valueParticle[i]:.6f}\n")  # Iteration und Objective Value

    print(f"Ergebnisse wurden automatisch in {filename} gespeichert.")
