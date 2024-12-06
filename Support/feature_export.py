import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd

def extraire_features(fichier_audio, start_time, end_time):
    # Calculer la durée à partir des temps de début et de fin
    duration = end_time - start_time
    # Charger le fichier audio entre les secondes spécifiées
    y, sr = librosa.load(fichier_audio, offset=start_time, duration=duration, sr=16000)
    

    
    # # Génération d'une sinus de 1kHz de 2s en fonction de sr
    # t = np.linspace(0, 2, 2*sr, endpoint=False)
    # y = 80*np.sin(2*np.pi*1000*t)
    

    # Calcule fenetre de hanning
    window = librosa.filters.get_window('hann', 1024, fftbins=True)
    # Obtenir les coefficients des filtres Mel
    mel_filters = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=30,norm=1.0)

    #separation des données en frames avec un overlap de 50%
    frames = librosa.util.frame(y, frame_length=1024, hop_length=512)[:, :32]

    # Application de la fenetre de hanning
    frames = frames * window[:, None]

    # calcul de la rfft des frames
    rfft = np.fft.rfft(frames, axis=0)

    # calcul la magnitude au carré de la rfft
    dsp = np.abs(rfft)**2/1024

    # calcul de la puissance des filtres mel
    mel_power = np.log(np.dot(mel_filters, dsp) + 1e-10)

    # z-score normalization
    z_score = ((mel_power - np.mean(mel_power)) / np.std(mel_power))
    print(z_score.shape)

    return z_score






 

# Exemple d'utilisation
fichier_audio = 'SAMP_000.WAV'
start_time = 0  # seconde de début
end_time = 2   # seconde de fin


features = extraire_features(fichier_audio, start_time, end_time)


with open('MEL_DATA.txt', 'r') as file:
    data = file.readlines()

# Convertir les données en une liste de nombres
mel_data = [float(line.strip()) for line in data]

# Définir les dimensions du spectrogramme de Mel
n_mels = 30  # Nombre de coefficients Mel
n_frames = 32  # Nombre de frames

# Convertir la liste en une matrice de taille (n_mels, n_frames)
mel_spectrogram = np.array(mel_data).reshape((n_frames,n_mels)).T

# Afficher le spectrogramme de Mel
plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
plt.title('Spectrogramme de Mel')
plt.xlabel('Frames')
plt.ylabel('Coefficients Mel')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()


# Plot du spectrograme de mel
plt.figure(figsize=(10, 4))
plt.imshow(features, aspect='auto', origin='lower', cmap='viridis')
plt.title('Spectrogramme de Mel')
plt.xlabel('Frames')
plt.ylabel('Coefficients Mel')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

