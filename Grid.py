import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah frame ke skala abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Membuat gambar biner dengan kriteria yang ditentukan
    binary_image = np.zeros_like(gray)  # Membuat gambar kosong dengan ukuran yang sama
    binary_image[gray > 30] = 255       # Piksel terang (>10) menjadi putih, yang <=10 tetap hitam

    # Tampilkan gambar asli dan gambar biner
    cv2.imshow('Binary Image', binary_image)

    # Keluar dari loop jika 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
