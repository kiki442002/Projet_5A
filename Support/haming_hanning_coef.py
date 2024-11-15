import numpy as np

def generate_hamming_window(size):
    # Calculer les coefficients de la fenêtre de Hamming
    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    return hamming_window

def print_hamming_window_as_c_array(hamming_window):
    # Afficher les coefficients sous forme de tableau en C
    print("float32_t hammingWindow[{}] = {{".format(len(hamming_window)))
    for i, coeff in enumerate(hamming_window):
        if i % 8 == 0:
            print("    ", end="")
        print("{:.6f}".format(coeff), end="")
        if i < len(hamming_window) - 1:
            print(", ", end="")
        if (i + 1) % 8 == 0:
            print()
    print("\n};")

def generate_hanning_window(size):
    # Calculer les coefficients de la fenêtre de Hanning
    hanning_window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    return hanning_window

def print_hanning_window_as_c_array(hanning_window):
    # Afficher les coefficients sous forme de tableau en C
    print("float32_t hanningWindow[{}] = {{".format(len(hanning_window)))
    for i, coeff in enumerate(hanning_window):
        if i % 8 == 0:
            print("    ", end="")
        print("{:.6f}".format(coeff), end="")
        if i < len(hanning_window) - 1:
            print(", ", end="")
        if (i + 1) % 8 == 0:
            print()
    print("\n};")

# Taille de la fenêtre de Hamming
window_size = 1024

# # Générer la fenêtre de Hamming
# hamming_window = generate_hamming_window(window_size)

# # Afficher les coefficients sous forme de tableau en C
# print_hamming_window_as_c_array(hamming_window)

# Générer la fenêtre de Hanning
hanning_window = generate_hanning_window(window_size)

# Afficher les coefficients sous forme de tableau en C
print_hanning_window_as_c_array(hanning_window)