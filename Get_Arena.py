import cv2
import numpy as np

# Fungsi untuk melakukan threshold dan mengubah gambar menjadi biner
def threshold_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return binary

# Fungsi untuk mendeteksi area kertas dan memotongnya
def detect_and_crop_paper(binary_image):
    # Temukan kontur dari gambar biner
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Asumsikan kontur terbesar adalah kertas
    contour = max(contours, key=cv2.contourArea)

    # Buat mask untuk area kertas
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Potong area kertas dari gambar biner
    paper_area = cv2.bitwise_and(binary_image, mask)

    # Luruskan area kertas
    return straighten_paper(contour, paper_area)

# Fungsi untuk meluruskan area kertas
def straighten_paper(contour, paper_area):
    # Mendapatkan 4 titik sudut dari kontur
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        # Mengurutkan titik-titik sudut
        points = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # Kiri atas
        rect[2] = points[np.argmax(s)]  # Kanan bawah

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # Kanan atas
        rect[3] = points[np.argmax(diff)]  # Kiri bawah

        # Mendapatkan ukuran baru
        widthA = np.linalg.norm(rect[1] - rect[0])
        widthB = np.linalg.norm(rect[2] - rect[3])
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(rect[3] - rect[0])
        heightB = np.linalg.norm(rect[2] - rect[1])
        maxHeight = max(int(heightA), int(heightB))

        # Menentukan titik tujuan
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

        # Melakukan perspektif transformasi
        M = cv2.getPerspectiveTransform(rect, dst)
        straightened = cv2.warpPerspective(paper_area, M, (maxWidth, maxHeight))

        return straightened

    return paper_area  # Jika tidak ada 4 sudut, kembalikan area kertas asli

# Fungsi untuk melakukan downsampling
def downsample_image(image, scale):
    # Downsample dengan pengambilan setiap `scale` pixel
    downsampled = image[::scale, ::scale]
    return downsampled

def main():
    # Baca gambar arena
    image = cv2.imread('image.png')  # Ganti dengan path gambar Anda

    if image is None:
        print("Gambar tidak ditemukan. Periksa path gambar.")
        return

    # Lakukan threshold untuk mengubah gambar menjadi biner
    binary_image = threshold_image(image)

    # Lakukan downsampling untuk mendapatkan resolusi 1:20
    scale = 20
    downsampled_image = downsample_image(binary_image, scale)

    # Dapatkan area kertas dan potong
    straightened_paper_area = detect_and_crop_paper(downsampled_image)

    # Tampilkan gambar biner dan area kertas yang sudah dipotong dan diluruskan
    cv2.imshow("Binary Image", binary_image)
    cv2.imshow("Straightened Paper Area", straightened_paper_area)

    # Print nilai setiap pixel dalam bentuk representasi biner
    print("Nilai setiap pixel dalam representasi biner:")
    for row in straightened_paper_area:
        print(' '.join('1' if pixel == 255 else '0' for pixel in row))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
