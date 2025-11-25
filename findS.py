#Verly Rahma Aulia
#2311501601

import numpy as np

#1. Dataset
data_list = [
    ["1","Pegawai","Tinggi","baik","Milik","Ya"],
    ["2","Pegawai","Tinggi","baik","Sewa","Ya"],
    ["3","Wirausaha","Sedang","baik","Milik","Tidak"],
    ["4","Pegawai","Sedang","buruk","Milik","Tidak"],
    ["5","Pegawai","Sedang","baik","Milik","Ya"],
    ["6","Wirausaha","Tinggi","baik","Sewa","Ya"],
    ["7","Wirausaha","Tinggi","buruk","Milik","Tidak"],   
]

#2. ubah ke numpy array
data = np.array(data_list, dtype=object)

#3. Tentukan indeks fitur dan label
fitur_idx = [1, 2, 3, 4]  # Indeks fitur
label_idx = 5  # Indeks label

#4. Filter contoh positif (pinjaman disetujui == "Ya")
positif = data[data[:, label_idx] == "Ya"]

#5. Inisialisasi hipotesis dengan contoh positif pertama (fitur saja)
hipotesis = list(positif[0, fitur_idx])

#6. Iterasi dan update hipotesis berdasarkan contoh positif berikutnya
for contoh in positif[1:]:
    for i, idx in enumerate(fitur_idx):
        if hipotesis[i] != contoh[idx]:
            hipotesis[i] = '?'  # Ganti dengan '?' jika tidak sama

#7. output hipotesis akhir
print("Hipotesis Akhir:", hipotesis)