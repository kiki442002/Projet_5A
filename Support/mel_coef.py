import librosa
import numpy as np
from scipy.sparse import csr_matrix

# Définir les paramètres pour les filtres Mel
n_mels = 30  # Nombre de coefficients Mel
n_fft = 1024  # Taille de la FFT
sr = 16000  # Fréquence d'échantillonnage

# Obtenir les coefficients des filtres Mel
mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

print(mel_filters[2])
print(sum(mel_filters[1]))
print(sum(mel_filters[2]))
print(sum(mel_filters[3]))
print(sum(mel_filters[4]))
print(sum(mel_filters[5]))
print(sum(mel_filters[6]))



# #import matplotlib.pyplot as plt

# # # Afficher les filtres Mel sur un graphique
# # # Calculer les fréquences correspondantes
# # frequencies = np.linspace(0, sr / 2, int(1 + n_fft // 2))

# # plt.figure(figsize=(10, 6))
# # for i in range(n_mels):
# #     plt.plot(frequencies,mel_filters[i])

# # plt.title('Filtres Mel')
# # plt.xlabel('Fréquence (Hz)')
# # plt.ylabel('Amplitude')
# # plt.grid(True)
# # plt.show()


# # Initialiser les tableaux
# non_zero_data = []
# zeros_before = []
# num_non_zero = []

# # Analyser les filtres Mel
# for i in range(n_mels):
#     filter_data = mel_filters[i]
#     non_zero_indices = np.nonzero(filter_data)[0]
    
#     if len(non_zero_indices) > 0:
#         zeros_before.append(non_zero_indices[0])
#         num_non_zero.append(len(non_zero_indices))
#         non_zero_data.extend(filter_data[non_zero_indices])
#     else:
#         zeros_before.append(len(filter_data))
#         num_non_zero.append(0)

# # Générer le fichier C optimisé
# with open('mel_filters_c_optimized.c', 'w') as f:
#     f.write('#include <stdio.h>\n\n')
#     f.write('const int n_mels = {};\n'.format(n_mels))
#     f.write('const int n_fft = {};\n'.format(n_fft))
#     f.write('const int non_zero_data_size = {};\n'.format(len(non_zero_data)))
    
#     f.write('const float mel_filters_non_zero_data[] = {\n')
#     f.write(', '.join(map(str, non_zero_data)))
#     f.write('\n};\n\n')

#     f.write('const int mel_filters_zeros_before[] = {\n')
#     f.write(', '.join(map(str, zeros_before)))
#     f.write('\n};\n\n')

#     f.write('const int mel_filters_num_non_zero[] = {\n')
#     f.write(', '.join(map(str, num_non_zero)))
#     f.write('\n};\n')

# print("Fichier C optimisé généré avec succès.")