import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definiere die Funktion
def objectiveFunction(u, v):
    return (u - 2)**2 + (v + 3)**2 + np.sin(u**2 + v**2)

# Erstelle ein Gitter von u- und v-Werten
u_values = np.linspace(-7, 7, 100)  # Werte für u
v_values = np.linspace(-7, 7, 100)  # Werte für v
u, v = np.meshgrid(u_values, v_values)  # Erstelle ein Gitter aus u und v

# Berechne die Funktionswerte
z_values = objectiveFunction(u, v)

# Erstelle den 3D-Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Erstelle die Oberfläche
ax.plot_surface(u, v, z_values, cmap='viridis')

# Füge Achsenbeschriftungen hinzu
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('f(u, v)')
ax.set_title('Plot der Funktion $f(u, v) = (u - 2)^2 + (v + 3)^2 + \sin(u^2 + v^2)$')

# Zeige den Plot
plt.show()
