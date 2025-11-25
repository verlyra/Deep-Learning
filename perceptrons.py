#Verly Rahma Aulia
#2311501601

import numpy as np

# Data untuk gerbang logika OR
data = [
    (+1, +1, +1),
    (+1, -1, +1),
    (-1, +1, +1),
    (-1, -1, -1)
]


# Inisialisasi bobot
w1 = 0
w2 = 0
wb = 0

alpha = 1 #laju pembelajaran
b = 1 #input bias
theta = 0.2 #ambang batas (threshold)
perubahan = True
epoch = 0

#fungsi aktivasi
def aktivasi(y_in, theta):
    if y_in >= theta:
        return 1
    elif -theta < y_in < theta:
        return 0
    else:
        return -1
    
#Proses pembelajaran
print("--> PROSES PEMBELAJARAN PERCEPTRON <--\n")
while perubahan:
    perubahan = False #tidak ada perubahan 
    epoch += 1
    print(f"\nEpoch: {epoch}")
    print("x1 x2 target y_in y dw1 dw2 dwb w1 w2 wb")

    for x1, x2, t in data:
        #Hitung tanggapan (y_in)
        y_in = (w1 * x1) + (w2 * x2) + (wb * b)


        #Tentukan output berdasarkan fungsi aktivasi
        y = aktivasi(y_in, theta)

        #Hitung perubahan bobot
        if y != t:
            dw1 = alpha * x1 * (t - y)
            dw2 = alpha * x2 * (t - y)
            dwb = alpha * b * (t - y)

            perubahan = True #ada perubahan bobot
            
            #Update bobot
            w1 += dw1
            w2 += dw2
            wb += dwb
        else:
            dw1 = 0
            dw2 = 0
            dwb = 0

        #Cetak hasil setiap langkah
        print(f"{x1:2} {x2:2}  {t:2} {y_in:5.2f} {y:2} {dw1:3} {dw2:3} {dwb:3} {w1:3} {w2:3} {wb:3}")

print("\n--> PEMBELAJARAN SELESAI <--")
print(f"Bobot akhir -> w1={w1}, w2={w2}, wb={wb}")
print(f"Total epoch: {epoch}")
