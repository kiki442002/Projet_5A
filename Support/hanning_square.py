import numpy as np
import matplotlib.pyplot as plt

# Définir la taille de la fenêtre et la fréquence d'échantillonnage
window_size = 30
fs = 16000  # Fréquence d'échantillonnage

# Générer un signal nul (ou un signal de bruit blanc)
signal =  np.ones(window_size)
# Appliquer une fenêtre de Hanning
hanning_window = np.hanning(window_size)
signal_hanning = signal * hanning_window

# Appliquer une fenêtre rectangulaire
rectangular_window = np.ones(window_size)
signal_rectangular = signal * rectangular_window

# Augmenter la résolution de la FFT en utilisant une taille de FFT plus grande
fft_size = 8192

# Calculer la FFT pour chaque fenêtre
fft_hanning = np.fft.fft(signal_hanning, n=fft_size)
fft_rectangular = np.fft.fft(signal_rectangular, n=fft_size)

# Calculer les fréquences correspondantes
frequencies = np.fft.fftfreq(fft_size, 1 / fs)

# Convertir les amplitudes en dB
fft_hanning_db = 20 * np.log10(np.abs(fft_hanning) + 1e-10)
fft_rectangular_db = 20 * np.log10(np.abs(fft_rectangular) + 1e-10)

# Utiliser np.fft.fftshift pour réorganiser les fréquences et les amplitudes
frequencies = np.fft.fftshift(frequencies)
fft_hanning_db = np.fft.fftshift(fft_hanning_db)
fft_rectangular_db = np.fft.fftshift(fft_rectangular_db)


# Afficher les résultats
plt.figure(figsize=(12, 8))

# Afficher la FFT avec la fenêtre de Hanning
plt.subplot(2, 1, 1)
plt.plot(frequencies, fft_hanning_db)
plt.title('FFT avec Fenêtre de Hanning')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude (dB)')
plt.grid(True)

# Afficher la FFT avec la fenêtre rectangulaire
plt.subplot(2, 1, 2)
plt.plot(frequencies, fft_rectangular_db)
plt.title('FFT avec Fenêtre Rectangulaire')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude (dB)')
plt.grid(True)

plt.tight_layout()
plt.show()