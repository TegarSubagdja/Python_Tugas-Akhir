import cv2
import numpy as np
import concurrent.futures

# Fungsi untuk melakukan threshold dan mengubah gambar menjadi biner
def threshold_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return binary

# Fungsi untuk mendeteksi area kertas dan memotongnya
def detect_and_crop_paper(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None  # Jika tidak ada kontur ditemukan, kembalikan None

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
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
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
    downsampled = image[::scale, ::scale]
    return downsampled

def main():
    # Inisialisasi webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam tidak dapat diakses.")
        return

    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            print("Gagal menangkap frame dari webcam.")
            break

        # Lakukan threshold untuk mengubah gambar menjadi biner
        binary_image = threshold_image(frame)

        # Lakukan downsampling untuk mendapatkan resolusi 1:20
        scale = 20
        downsampled_image = downsample_image(binary_image, scale)

        # Menggunakan multithreading untuk mendeteksi dan memotong area kertas
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(detect_and_crop_paper, downsampled_image)
            straightened_paper_area = future.result()

        # Tampilkan gambar biner
        cv2.imshow("Binary Image", binary_image)

        # Tampilkan area kertas yang sudah dipotong dan diluruskan
        if straightened_paper_area is not None:
            cv2.imshow("Straightened Paper Area", straightened_paper_area)
        else:
            cv2.imshow("Straightened Paper Area", np.zeros_like(frame))  # Tampilkan gambar hitam jika tidak ada area yang terdeteksi

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan semua sumber daya
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
