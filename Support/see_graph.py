import matplotlib.pyplot as plt
import numpy as np
import librosa
import math

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


def verify_data(dse, mel):
    # Lire les données du fichier HANN.TXT
    with open(dse, 'r') as file:
        data = file.readlines()
    # Convertir les données en une liste de nombres
    dseValue = np.array([float(line.strip()) for line in data])/1024

    # Lire les données du fichier HANN.TXT
    with open(mel, 'r') as file:
        data = file.readlines()
    # Convertir les données en une liste de nombres
    melValue = [float(line.strip()) for line in data]
    print(melValue)

    # Définir les paramètres pour les filtres Mel
    n_mels = 30  # Nombre de coefficients Mel
    n_fft = 1024  # Taille de la FFT
    sr = 16000  # Fréquence d'échantillonnage

    # Obtenir les coefficients des filtres Mel
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,norm=1.0)

    # Appliquer les filtres Mel sur la DSE
    for i in range(n_mels):
        print(math.log(sum((np.array(mel_filters[i]))* dseValue)+1e-10))
    #print(melValue)




#plot_data('DSE.TXT', 'Visualisation d\'une DSE calculé sur une STM32', 'Fréquence (Hz)', 'Amplitude')
verify_data('DSE.TXT', 'MEL.TXT')