import matplotlib.pyplot as plt
import numpy as np

sr=16000


def plot_data(filename, title, x_label='Index', y_label='Valeur'):
    # Lire les données du fichier HANN.TXT
    with open(filename, 'r') as file:
        data = file.readlines()

    # Convertir les données en une liste de nombres
    numbers = [float(line.strip()) for line in data]

    #tableau des fréquences
    frequencies = np.linspace(0, sr / 2, len(numbers))

    # Créer un graphique
    plt.plot(frequencies,numbers)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    # Afficher le graphique
    plt.show()

plot_data('DSE.TXT', 'Visualisation d\'une DSE calculé sur une STM32', 'Fréquence (Hz)', 'Amplitude')