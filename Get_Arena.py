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
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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
        
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def get_corners(contour):
    """Approximate contour to get corner points."""
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) != 4:
        raise ValueError("Could not find exactly 4 corners")
        
    # Convert to more usable format
    corners = np.float32([point[0] for point in approx])
    
    # Sort corners: top-left, top-right, bottom-right, bottom-left
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
    # Calculate width and height for the new image
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
    
    # Define destination points for transform
    dst_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Calculate and apply perspective transform
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    straightened = cv2.warpPerspective(img, matrix, (max_width, max_height))
    
    return straightened

def downsample_image(img, scale_factor):
    """Downsample image by the specified scale factor."""
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    return cv2.resize(img, (width, height))

def process_document(image_path, scale_factor=0.5, visualize=True):
    """Main function to process the document image."""
    # Load and process image
    original = load_image(image_path)
    binary = convert_to_binary(original)
    largest_contour = detect_contours(binary)
    corners = get_corners(largest_contour)
    straightened = straighten_paper(original, corners)
    
    # Downsample if requested
    if scale_factor != 1.0:
        straightened = downsample_image(straightened, scale_factor)
    
    # Visualize results if requested
    if visualize:
        # Draw contour on original image
        img_with_contours = original.copy()
        cv2.drawContours(img_with_contours, [largest_contour], -1, (0, 255, 0), 2)
        
        # Create visualization
        cv2.imshow('Original Image', original)
        cv2.imshow('Binary Image', binary)
        cv2.imshow('Detected Contours', img_with_contours)
        cv2.imshow('Straightened Document', straightened)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return straightened

if __name__ == "__main__":
    # Example usage
    image_path = "image.png"  # Replace with actual image path
    try:
        result = process_document(image_path)
        print("\nBinary representation of the processed document:")
    except Exception as e:
        print(f"Error processing document: {str(e)}")