import tensorflow as tf
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_path = "app/model/model_vgg16.keras"
        self.class_names = ['Cataract', 'Normal']
        
    def load_model(self):
        """Download (if needed) and load the VGG16 .keras model"""
        try:
            self._download_model_if_needed()

            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f" VGG16 model loaded from {self.model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")

            self.model.summary()
            self._test_model()

        except Exception as e:
            logger.error(f" Error loading VGG16 model: {str(e)}")
            raise e

    def _download_model_if_needed(self):
        """Download model from Google Drive if not already present"""
        if not os.path.exists(self.model_path):
            try:
                import gdown
                url = "https://drive.google.com/uc?id=1VweyVdVK4CULSNLQKQ-aK8b541In4eAj&export=download"
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                logger.info(" Downloading model_vgg16.keras from Google Drive...")
                gdown.download(url, self.model_path, quiet=False)
                if os.path.exists(self.model_path):
                    logger.info(" Model successfully downloaded.")
                else:
                    logger.error(" Model download failed.")
            except Exception as e:
                logger.error(f" Failed to download model: {e}")
                raise e
        else:
            logger.info(" Model already exists locally.")

    def _test_model(self):
        try:
            input_shape = self.model.input_shape[1:]
            dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
            dummy_input *= 255.0
            prediction = self.model.predict(dummy_input, verbose=0)
            logger.info(f"VGG16 test prediction shape: {prediction.shape}")
            logger.info(f"Sample prediction: {prediction}")
        except Exception as e:
            logger.warning(f" VGG16 model test failed: {e}")

    def predict(self, processed_image: np.ndarray):
        if self.model is None:
            raise ValueError(" Model not loaded. Call load_model() first.")

        try:
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)

            predictions = self.model.predict(processed_image, verbose=1)
            logger.info(f" Raw prediction: {predictions}")

            if predictions.shape[-1] == 1:
                raw_confidence = float(predictions[0][0])
                predicted_class = 1 if raw_confidence > 0.5 else 0
                final_confidence = raw_confidence if predicted_class == 1 else 1 - raw_confidence
                all_probs = [1 - raw_confidence, raw_confidence]
            else:
                predicted_class = int(np.argmax(predictions[0]))
                final_confidence = float(np.max(predictions[0]))
                all_probs = predictions[0].tolist()

            return {
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': final_confidence,
                'all_probabilities': all_probs,
                'raw_prediction': predictions[0].tolist(),
                'model_type': 'VGG16'
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise e

    def get_model_info(self):
        if self.model is None:
            return {
                'loaded': False,
                'model_type': 'VGG16',
                'class_names': self.class_names
            }

        try:
            return {
                'loaded': True,
                'path': self.model_path,
                'model_type': 'VGG16 Keras',
                'class_names': self.class_names,
                'input_shape': str(self.model.input_shape),
                'output_shape': str(self.model.output_shape),
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'layers_count': len(self.model.layers),
                'last_layer_activation': self.model.layers[-1].activation.__name__
            }

        except Exception as e:
            logger.error(f" Error getting model info: {str(e)}")
            return {'error': str(e)}

# Global instance
model_loader = ModelLoader()

def get_model():
    if model_loader.model is None:
        model_loader.load_model()
    return model_loader
