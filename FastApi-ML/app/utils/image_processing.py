import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_data, target_size=(224, 224)):
    """
    Preprocess image for VGG16 cataract detection model
    VGG16 typically expects specific preprocessing
    """
    try:
        # Convert bytes to PIL Image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Original image size: {image.size}")
        logger.info(f"Original image mode: {image.mode}")
        
        # Resize image to VGG16 input size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        logger.info(f"Processed image shape: {img_array.shape}")
        logger.info(f"Pixel value range before normalization: {img_array.min()} - {img_array.max()}")
        
        # VGG16 preprocessing: normalize to [0, 1]
        img_array = img_array / 255.0
        
        logger.info(f"Final pixel range: {img_array.min()} - {img_array.max()}")
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image for VGG16: {str(e)}")
        raise e

class ImageProcessor:
    """Image processor class for VGG16 model"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def process(self, image_data):
        """Process image data"""
        return preprocess_image(image_data, self.target_size)

# Global image processor instance
_image_processor = None

def get_image_processor():
    """Get the global image processor instance"""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor

# For backward compatibility
def process_image(image_data, target_size=(224, 224)):
    """Process image - backward compatibility function"""
    return preprocess_image(image_data, target_size)
