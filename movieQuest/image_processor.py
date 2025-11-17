import cv2
import numpy as np
import easyocr
from PIL import Image
import requests
from io import BytesIO

# Initialize EasyOCR reader (only once, globally)
# This will download models on first use (~80MB) but won't require system dependencies
reader = None

def get_reader():
    """Lazy load the EasyOCR reader"""
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    return reader


def detect_and_blur_text(image_url, blur_strength=25):
    """
    Download an image, detect text regions using OCR, and remove them using inpainting.
    
    Args:
        image_url: URL of the image to process
        blur_strength: Not used for inpainting, kept for compatibility
    
    Returns:
        Processed image as numpy array
    """
    try:
        print(f"[OCR] Starting to process: {image_url}")
        
        # Download the image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Convert to PIL Image
        img_pil = Image.open(BytesIO(response.content))
        print(f"[OCR] Image downloaded, size: {img_pil.size}")
        
        # Convert PIL to OpenCV format (RGB to BGR)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Create a copy to work with
        result = img.copy()
        
        # Create a mask for inpainting (white = areas to inpaint)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Get EasyOCR reader
        print(f"[OCR] Initializing EasyOCR reader...")
        ocr_reader = get_reader()
        print(f"[OCR] Reader initialized, detecting text...")
        
        # Detect text using EasyOCR
        # Returns list of (bbox, text, confidence)
        detections = ocr_reader.readtext(img)
        
        print(f"[OCR] Found {len(detections)} text regions")
        
        # Process each detected text region
        inpaint_count = 0
        for (bbox, text, confidence) in detections:
            print(f"[OCR] Detected: '{text}' (confidence: {confidence:.2f})")
            
            # Only process if confidence is above threshold
            if confidence > 0.3:  # 30% confidence threshold
                # bbox is a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Get bounding rectangle
                bbox = np.array(bbox).astype(int)
                x_min = max(0, bbox[:, 0].min())
                y_min = max(0, bbox[:, 1].min())
                x_max = min(img.shape[1], bbox[:, 0].max())
                y_max = min(img.shape[0], bbox[:, 1].max())
                
                # Add padding around text region
                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(img.shape[1], x_max + padding)
                y_max = min(img.shape[0], y_max + padding)
                
                # Draw white rectangle on mask for this text region
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
                inpaint_count += 1
                print(f"[OCR] Marked text region for removal at ({x_min},{y_min}) to ({x_max},{y_max})")
        
        # Perform inpainting if we found any text
        if inpaint_count > 0:
            print(f"[OCR] Inpainting {inpaint_count} text regions...")
            # Use Telea inpainting algorithm (fast and good for text removal)
            result = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            print(f"[OCR] Inpainting complete!")
        else:
            print(f"[OCR] No text regions to remove.")
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        # If processing fails, return the original image
        try:
            response = requests.get(image_url, timeout=10)
            img_pil = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except:
            return None


def save_processed_image(image_array, output_path):
    """
    Save the processed image to disk.
    
    Args:
        image_array: Numpy array of the image (BGR format)
        output_path: Path where to save the image
    """
    if image_array is not None:
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the image
            success = cv2.imwrite(output_path, image_array)
            if success:
                print(f"[SAVE] Image saved successfully to: {output_path}")
                # Verify file exists
                if os.path.exists(output_path):
                    print(f"[SAVE] File verified, size: {os.path.getsize(output_path)} bytes")
                    return True
                else:
                    print(f"[SAVE] ERROR: File was not created!")
                    return False
            else:
                print(f"[SAVE] ERROR: cv2.imwrite returned False")
                return False
        except Exception as e:
            print(f"[SAVE] ERROR saving image: {e}")
            return False
    return False
