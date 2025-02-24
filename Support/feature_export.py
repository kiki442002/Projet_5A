import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import cmsisdsp as dsp

def generate_hanning_window(size):
    # Calculer les coefficients de la fenêtre de Hanning
    hanning_window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    return hanning_window

def extraire_features(fichier_audio, start_time, end_time):
    # Calculer la durée à partir des temps de début et de fin
    duration = end_time - start_time
    # Charger le fichier audio entre les secondes spécifiées
    y, sr = librosa.load(fichier_audio, offset=start_time, duration=duration, sr=16000)
    

    
    # # Génération d'une sinus de 1kHz de 2s en fonction de sr
    # t = np.linspace(0, 2, 2*sr, endpoint=False)
    # y = 80*np.sin(2*np.pi*1000*t)
    

    # Calcule fenetre de hanning
    window = generate_hanning_window(1024)
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

    # Normalisation Z-score pour chaque colonne
    mean = np.mean(mel_power, axis=0)
    std = np.std(mel_power, axis=0)
    mel_power = (mel_power - mean) / std

    # # Normalisation Min-Max pour chaque colonne
    # min_val = np.min(mel_power, axis=0)
    # max_val = np.max(mel_power, axis=0)
    # mel_power = (mel_power - min_val) / (max_val - min_val)

    return mel_power

def extraire_features_dsp(fichier_audio, start_time, end_time):
    # Calculer la durée à partir des temps de début et de fin
    duration = end_time - start_time
    # Charger le fichier audio entre les secondes spécifiées
    y, sr = librosa.load(fichier_audio, offset=start_time, duration=duration, sr=16000)
    

    
    # # Génération d'une sinus de 1kHz de 2s en fonction de sr
    # t = np.linspace(0, 2, 2*sr, endpoint=False)
    # y = 80*np.sin(2*np.pi*1000*t)
    

    # Calcule fenetre de hanning
    window = generate_hanning_window(1024)
    # Obtenir les coefficients des filtres Mel
    mel_filters = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=30,norm=1.0)

    #separation des données en frames avec un overlap de 50%
    frames = librosa.util.frame(y, frame_length=1024, hop_length=512)[:, :32]

    # Application de la fenetre de hanning
    frames = frames * window[:, None]
    frames = frames.T
    print(frames.shape)
   
    # Préparer un tableau pour stocker les résultats de la DSP
    dsp_output = np.zeros((frames.shape[0], 513), dtype=np.float32)

    # Initialiser la structure RFFT
    S = dsp.arm_rfft_fast_instance_f32()
    dsp.arm_rfft_fast_init_f32(S, 1024)
    
    for i,frame in enumerate(frames):
        frame_flat = frame.astype(np.float32)
        fft_result = np.zeros(1024, dtype=np.float32)
        fft_result=dsp.arm_rfft_fast_f32(S, frame_flat, 0)

         # Calculer la magnitude au carré des coefficients de la FFT pour obtenir la DSP
        dsp_result = np.zeros(513, dtype=np.float32)
        dsp_result[1:-1]=dsp.arm_cmplx_mag_squared_f32(fft_result[2:])
        dsp_result[0] = fft_result[0]**2
        dsp_result[-1] = fft_result[1]**2
        dsp_output[i] = dsp_result/1024
  

    # calcul de la puissance des filtres mel
    mel_power = np.log(np.dot(dsp_output, mel_filters.T) + 1e-10, dtype=np.float32)

    # z-score normalization
    for i,mel in enumerate(mel_power):
        mean = dsp.arm_mean_f32(mel)
        std = dsp.arm_std_f32(mel)
        mel_power[i] = (mel - mean) / std
    #  # Normalisation Min-Max pour chaque colonne
    # min_val = np.min(mel_power, axis=0)
    # max_val = np.max(mel_power, axis=0)
    # mel_power = (mel_power - min_val) / (max_val - min_val)
    return mel_power




 

# Exemple d'utilisation
fichier_audio = 'SAMP_000.WAV'
start_time = 0  # seconde de début
end_time = 2   # seconde de fin


features1 = extraire_features(fichier_audio, start_time, end_time)
# features2 = extraire_features_dsp(fichier_audio, start_time, end_time)
# np.savetxt('features2.txt', features2, delimiter='\n', fmt='%.6f')

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
plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis', vmin=-2, vmax=5)
plt.title('Spectrogramme de Mel sur STM32')
plt.xlabel('Frames')
plt.ylabel('Coefficients Mel')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()


# with open('DATA_IN.txt', 'r') as file:
#     data = file.readlines()

# # Convertir les données en une liste de nombres
# data_in = [float(line.strip()) for line in data]

# # Définir les dimensions du spectrogramme de Mel
# n_mels = 30  # Nombre de coefficients Mel
# n_frames = 32  # Nombre de frames

# # Convertir la liste en une matrice de taille (n_mels, n_frames)
# mel_spectrogram_in = np.array(data_in).reshape((n_mels,n_frames))

# # Afficher le spectrogramme de Mel
# plt.figure(figsize=(10, 4))
# plt.imshow(mel_spectrogram_in, aspect='auto', origin='lower', cmap='viridis', vmin=-2, vmax=5)
# plt.title('Spectrogramme de Mel stm32')
# plt.xlabel('Frames')
# plt.ylabel('Coefficients Mel')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()



# Plot du spectrograme de mel
plt.figure(figsize=(10, 4))
plt.imshow(features1, aspect='auto', origin='lower', cmap='viridis', vmin=-2, vmax=5)
plt.title('Spectrogramme de Mel sur PC')
plt.xlabel('Frames')
plt.ylabel('Coefficients Mel')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()



# plt.figure(figsize=(10, 4))
# plt.imshow(features2.T, aspect='auto', origin='lower', cmap='viridis', vmin=-2, vmax=5)
# plt.title('Spectrogramme de Mel features2')
# plt.xlabel('Frames')
# plt.ylabel('Coefficients Mel')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()

# plt.figure(figsize=(10, 4))
# plt.imshow(features2.T-features1, aspect='auto', origin='lower', cmap='viridis', vmin=-2, vmax=5)
# plt.title('Spectrogramme de Mel fdif features')
# plt.xlabel('Frames')
# plt.ylabel('Coefficients Mel')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()




plt.show()

