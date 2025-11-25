import numpy as np

# 1. Data input dan target (fungsi logika AND)
x = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

t = np.array([1, -1, -1, -1])  # Target output

# 2. Parameter awal
alpha = 0.1      # learning rate
epoch = 100      # batas maksimum iterasi
epsilon = 1e-6   # ambang perubahan bobot kecil (konvergen)
np.random.seed(0)  # agar hasil acak konsisten

# 3. Inisialisasi bobot (termasuk bias)
w = np.random.uniform(-0.5, 0.5, x.shape[1])  # w1, w2
b = np.random.uniform(-0.5, 0.5)              # bias

print(f"Bobot awal: {w}, b: {b:.4f}\n")

# 4. Proses Pelatihan (Adaline)
for epoch in range(epoch):
    total_change = 0

    for i in range(len(x)):
        xi = x[i]
        ti = t[i]

        # Hitung output linear
        y_in = np.dot(w, xi) + b

        # Hitung error
        error = ti - y_in

        # Hitung perubahan bobot
        delta_w = alpha * error * xi
        delta_b = alpha * error

        # Update bobot dan bias
        w += delta_w
        b += delta_b

        # Catat total perubahan (untuk deteksi konvergensi)
        total_change += np.abs(delta_w.sum()) + abs(delta_b)

    print(f"Epoch {epoch+1:2d} | Bobot: {w}, Bias: {b:.4f}, Total Perubahan: {total_change:.6f}")

    # Kriteria berhenti (konvergen)
    if total_change < epsilon:
        print("\nTraining selesai (konvergen).")
        break

# 5. Pengujian Hasil
print("\n=== Pengujian ===")
for i in range(len(x)):
    xi = x[i]
    y_in = np.dot(w, xi) + b
    y_out = 1 if y_in > 0 else -1  # fungsi aktivasi biner
    print(f"Input: {xi}, y_in: {y_in:.4f}, Output: {y_out}, Target: {t[i]}")
