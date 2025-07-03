"""
Image processing utilities for garbage detection
"""
import numpy as np
from PIL import Image
import io

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image object
        target_size: tuple of (width, height) for resizing
    
    Returns:
        numpy array: preprocessed image array
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values (0-1 range)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def validate_image(file):
    """
    Validate uploaded image file
    
    Args:
        file: uploaded file object
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check file size (max 16MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 16 * 1024 * 1024:  # 16MB
            return False, "File terlalu besar. Maksimal 16MB."
        
        # Try to open as image
        image = Image.open(file)
        
        # Check image dimensions
        width, height = image.size
        if width < 32 or height < 32:
            return False, "Gambar terlalu kecil. Minimal 32x32 pixels."
        
        if width > 4096 or height > 4096:
            return False, "Gambar terlalu besar. Maksimal 4096x4096 pixels."
        
        return True, None
        
    except Exception as e:
        return False, f"File bukan gambar yang valid: {str(e)}"

def get_image_info(image):
    """
    Get information about the image
    
    Args:
        image: PIL Image object
    
    Returns:
        dict: image information
    """
    return {
        'format': image.format,
        'mode': image.mode,
        'size': image.size,
        'width': image.width,
        'height': image.height
    }