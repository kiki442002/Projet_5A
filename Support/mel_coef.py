import librosa
import numpy as np


# Définir les paramètres pour les filtres Mel
n_mels = 30  # Nombre de coefficients Mel
n_fft = 1024  # Taille de la FFT
sr = 16000  # Fréquence d'échantillonnage

# Obtenir les coefficients des filtres Mel
mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,norm=1.0)


def plot_mel_filters(mel_filters):
    import matplotlib.pyplot as plt

    # Afficher les filtres Mel sur un graphique
    # Calculer les fréquences correspondantes
    frequencies = np.linspace(0, sr / 2, int(1 + n_fft // 2))

    plt.figure(figsize=(10, 6))
    for i in range(n_mels):
        plt.plot(frequencies,mel_filters[i])

    plt.title('Filtres Mel')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def mel_filters_to_c(mel_filters):
    # Initialiser les tableaux
    non_zero_data = []
    zeros_before = []
    num_non_zero = []

    # Analyser les filtres Mel
    for i in range(n_mels):
        filter_data = mel_filters[i]
        non_zero_indices = np.nonzero(filter_data)[0]

        if len(non_zero_indices) > 0:
            zeros_before.append(non_zero_indices[0])
            num_non_zero.append(len(non_zero_indices))
            non_zero_data.extend(filter_data[non_zero_indices])
        else:
            zeros_before.append(len(filter_data))
            num_non_zero.append(0)
    
    non_zero_data = list(np.array(non_zero_data)/n_fft)
    # Générer le fichier C optimisé
    with open('mel_filters.h', 'w') as f:
        f.write('#define N_MELS {}\n\n'.format(n_mels))

        f.write('const float32_t mel_filters_non_zero_data[] = {\n')
        f.write(', '.join(map(str, non_zero_data)))
        f.write('\n};\n\n')

        f.write('const int16_t mel_filters_zeros_before[] = {\n')
        f.write(', '.join(map(str, zeros_before)))
        f.write('\n};\n\n')

        f.write('const uint8_t mel_filters_num_non_zero[] = {\n')
        f.write(', '.join(map(str, num_non_zero)))
        f.write('\n};\n')

        print("Fichier C optimisé généré avec succès.")


# Afficher les filtres Mel
#plot_mel_filters(mel_filters)

# Générer le fichier C optimisé
mel_filters_to_c(mel_filters)

