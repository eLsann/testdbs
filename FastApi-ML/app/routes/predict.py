from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
from app.models.model_loader import get_model
from app.utils.image_processing import get_image_processor, preprocess_image
from app.schemas.prediction import PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse)
async def predict_cataract(file: UploadFile = File(...)):
    """
    Predict cataract from uploaded eye image using VGG16 model
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        logger.info(f"Received image: {file.filename}, size: {len(image_data)} bytes")
        
        # Process image
        image_processor = get_image_processor()
        processed_image = image_processor.process(image_data)
        
        # Alternative: use the direct function
        # processed_image = preprocess_image(image_data)
        
        logger.info(f"Image processed successfully, shape: {processed_image.shape}")
        
        # Load model and make prediction
        model = get_model()
        result = model.predict(processed_image)
        
        logger.info(f"Prediction completed: {result}")
        
        return PredictionResponse(
            success=True,
            predicted_class=result['predicted_class'],
            class_name=result['class_name'],
            confidence=result['confidence'],
            all_probabilities=result['all_probabilities'],
            message=f"Prediction: {result['class_name']} with {result['confidence']:.2%} confidence"
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded VGG16 model"""
    try:
        model = get_model()
        info = model.get_model_info()
        return JSONResponse(content=info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        model_info = model.get_model_info()
        
        return {
            "status": "healthy",
            "model_loaded": model_info.get('loaded', False),
            "model_type": model_info.get('model_type', 'Unknown')
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
