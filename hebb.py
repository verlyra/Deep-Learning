#Verly Rahma Aulia
#2311501601

import numpy as np

# Data input dengan bias (x0 = 1)
X = np.array([
    [1, 1, 1],   # bias, x1, x2
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1]
])

# Target output untuk logika OR (1 dan -1)
y = np.array([1, 1, 1, -1])

# Learning rate
eta = 1

# Inisialisasi bobot dengan nol (bias, w1, w2)
w = np.zeros(X.shape[1])

print("Bobot awal:", w)

# Proses update bobot
for i in range(len(y)):
    delta_w = eta * y[i] * X[i]   # Hitung perubahan bobot
    w += delta_w                  # Update bobot
    print(f"\nIterasi {i+1}")
    print("Input:", X[i])
    print("Target:", y[i])
    print("Perubahan bobot:", delta_w)
    print("Bobot baru:", w)

# Fungsi prediksi setelah pelatihan
def predict(x, w):
    return 1 if np.dot(w, x) > 0 else -1

# Tes prediksi semua input
print("\nPrediksi hasil:")
for i in range(len(X)):
    print(f"Input: {X[i][1:]}, Prediksi: {predict(X[i], w)}, Target: {y[i]}")
