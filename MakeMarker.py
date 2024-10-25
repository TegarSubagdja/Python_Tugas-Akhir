import cv2

# Cek apakah modul aruco tersedia
if hasattr(cv2, 'aruco'):
    print("Modul aruco tersedia!")
else:
    print("Modul aruco tidak ditemukan.")
