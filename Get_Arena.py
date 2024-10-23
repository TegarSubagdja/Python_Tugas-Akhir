import cv2
import numpy as np

def load_image(image_path):
    """Load image from the specified path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image from path")
    return img

def convert_to_binary(img):
    """Convert image to binary using adaptive thresholding."""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return binary

def detect_contours(binary):
    """Detect and return the largest contour (assumed to be the paper)."""
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        raise ValueError("No contours found in image")
        
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def get_corners(contour):
    """Approximate contour to get corner points."""
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) != 4:
        raise ValueError("Could not find exactly 4 corners")
        
    corners = np.float32([point[0] for point in approx])
    
    # Sort corners
    sum_pts = corners.sum(axis=1)
    diff_pts = np.diff(corners, axis=1)
    
    ordered_corners = np.zeros((4, 2), dtype=np.float32)
    ordered_corners[0] = corners[np.argmin(sum_pts)]    # Top-left
    ordered_corners[1] = corners[np.argmin(diff_pts)]   # Top-right
    ordered_corners[2] = corners[np.argmax(sum_pts)]    # Bottom-right
    ordered_corners[3] = corners[np.argmax(diff_pts)]   # Bottom-left
    
    return ordered_corners

def straighten_paper(img, corners):
    """Apply perspective transform to straighten the paper."""
    width_1 = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + 
                      ((corners[1][1] - corners[0][1]) ** 2))
    width_2 = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + 
                      ((corners[2][1] - corners[3][1]) ** 2))
    max_width = max(int(width_1), int(width_2))
    
    height_1 = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) + 
                       ((corners[3][1] - corners[0][1]) ** 2))
    height_2 = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) + 
                       ((corners[2][1] - corners[1][1]) ** 2))
    max_height = max(int(height_1), int(height_2))
    
    dst_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    straightened = cv2.warpPerspective(img, matrix, (max_width, max_height))
    
    return straightened

def downsample_image(img, target_ratio=10):
    """
    Downsample image by a target ratio (e.g., 1:10).
    Returns both the downsampled image and its binary version.
    """
    scale_factor = 1.0 / target_ratio
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    
    # Ensure minimum dimensions
    new_width = max(new_width, 1)
    new_height = max(new_height, 1)
    
    # Downsample using INTER_AREA for better quality
    downsampled = cv2.resize(img, (new_width, new_height), 
                            interpolation=cv2.INTER_AREA)
    
    # Convert downsampled image to binary
    binary_downsampled = convert_to_binary(downsampled)
    
    print(f"\nDownsampling Results:")
    print(f"Original size: {img.shape[1]}x{img.shape[0]}")
    print(f"Downsampled size: {downsampled.shape[1]}x{downsampled.shape[0]}")
    print(f"Actual ratio achieved: 1:{img.shape[1]/downsampled.shape[1]:.2f}")
    
    return downsampled, binary_downsampled

def print_binary_matrix(binary_img):
    """
    Print binary image as a matrix of 1s and 0s.
    """
    # Ensure we have a binary image (0s and 1s)
    binary = np.where(binary_img > 128, 1, 0)
    
    print(f"\nBinary Matrix ({binary.shape[0]}x{binary.shape[1]}):")
    print("   " + "".join([f"{i:3}" for i in range(binary.shape[1])]))  # Column numbers
    for i, row in enumerate(binary):
        print(f"{i:2} " + " ".join(map(str, row)))  # Row number + data

def process_document(image_path, target_ratio=10, visualize=True):
    """
    Process document image and return binary representation of downsampled image.
    """
    # Load and process image
    original = load_image(image_path)
    initial_binary = convert_to_binary(original)
    largest_contour = detect_contours(initial_binary)
    corners = get_corners(largest_contour)
    straightened = straighten_paper(original, corners)
    
    # Downsample and get binary version
    downsampled, binary_downsampled = downsample_image(straightened, target_ratio)
    
    # Visualize results if requested
    if visualize:
        img_with_contours = original.copy()
        cv2.drawContours(img_with_contours, [largest_contour], -1, (0, 255, 0), 2)
        
        cv2.imshow('Original', original)
        cv2.imshow('Initial Binary', initial_binary)
        cv2.imshow('Detected Contours', img_with_contours)
        cv2.imshow('Straightened', straightened)
        cv2.imshow('Downsampled', downsampled)
        cv2.imshow('Final Binary', binary_downsampled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return binary_downsampled

if __name__ == "__main__":
    # Example usage
    image_path = "image.png"  # Replace with actual image path
    target_ratio = 10  # For 1:10 ratio
    
    try:
        # Process document and get binary result
        binary_result = process_document(image_path, target_ratio=target_ratio)
        
        # Print binary matrix with row and column numbers
        print_binary_matrix(binary_result)
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")