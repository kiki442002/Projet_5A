import matplotlib.pyplot as plt
import numpy as np


def plot_data(filename, title):
    # Lire les données du fichier HANN.TXT
    with open(filename, 'r') as file:
        data = file.readlines()

    # Convertir les données en une liste de nombres
    numbers = [int(line.strip()) for line in data]

    # Créer un graphique
    plt.plot(numbers)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Valeur')
    plt.grid(True)

    # Afficher le graphique
    plt.show()

plot_data('HANN_COS.TXT', 'Visualisation de la porte de Hanning avec sinus à 1kHz')