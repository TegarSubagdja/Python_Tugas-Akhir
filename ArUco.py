import cv2
import numpy as np

def detect_marker_position():
    """
    Mendeteksi posisi 3D (x, y, z) dan orientasi yaw dari marker ArUco
    """
    cap = cv2.VideoCapture(0)
    
    # Parameter kamera
    focal_length = 1000.0
    center = (640, 360)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1))
    marker_size = 0.1  # 10 cm
    
    # Inisialisasi ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error membaca frame dari webcam")
                break
            
            # Deteksi marker
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                
                for i in range(len(ids)):
                    # Estimasi pose
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], marker_size, camera_matrix, dist_coeffs
                    )
                    
                    # Ambil posisi
                    x, y, z = tvecs[0][0]
                    
                    # Hitung yaw (rotasi pada sumbu-z)
                    rmat, _ = cv2.Rodrigues(rvecs[0])
                    _, _, yaw = cv2.RQDecomp3x3(rmat)[0]
                    
                    # Tampilkan informasi
                    info = f"X: {x:.3f} Y: {y:.3f} Z: {z:.3f} Yaw: {yaw:.1f}"
                    cv2.putText(frame, info, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Marker Position', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_marker_position()