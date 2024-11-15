import numpy as np
import matplotlib.pyplot as plt

# Definiere die Funktion
def objectiveFunction(x):
    return (x - 3)**2 + 1

# Erstelle ein Array von x-Werten für den Plot
x_values = np.linspace(-7, 13, 100)  # 100 Punkte für eine glattere Kurve
y_values = objectiveFunction(x_values)  # Berechne die Funktionswerte

# Erstelle den Plot
plt.plot(x_values, y_values, label=r"$f(x) = (x - 3)^2 + 1$")

# Füge Achsenbeschriftungen und Titel hinzu
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot der Funktion $f(x) = (x - 3)^2 + 1$')

# Setze die x-Achsen-Ticks mit einem Abstand von 1
plt.xticks(np.arange(np.floor(min(x_values)), np.ceil(max(x_values)) + 1, 1))

# Setze die y-Achsen-Ticks mit einem Abstand von 1

# Zeige das Gitter an
plt.grid()

# Zeige die Legende an
plt.legend()

# Zeige den Plot
plt.show()
