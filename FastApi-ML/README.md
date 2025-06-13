# Machine Learning API

This project implements a machine learning API using FastAPI (or Flask) that processes images and returns detection results. The API is designed to handle image uploads, perform predictions using a pre-trained model, and return the results in a structured format.

## Project Structure

```
ml_api_app
├── app
│   ├── main.py                # Entry point for the application
│   ├── models
│   │   └── model_loader.py    # Model loading and initialization
│   ├── routes
│   │   └── predict.py         # API endpoint for predictions
│   ├── utils
│   │   └── main.py # Image processing utilities
│   └── schemas
│       └── prediction.py      # Data schemas for requests and responses
|   └── model
|       └── model_vgg16.keras
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── tfjs_model_cnn
    ├── group1-shard1of1.bin   # Model weights
    ├── model.json             # Model architecture
    ├── model_custom_cnn.h5    # Keras model file
    └── model_custom_cnn.keras  # Alternative Keras model format
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd ml_api_app
   ```

2. **Install dependencies:**
   Create a virtual environment and install the required packages listed in `requirements.txt`.
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Start the FastAPI (or Flask) server.
   ```
   uvicorn app.main:app --reload
   ```

## Usage

### Making Predictions

To make a prediction, send a POST request to the `/predict` endpoint with an image file. The request should include the image in the form-data.

**Example using `curl`:**
```
curl -X POST "http://localhost:8000/predict" -F "file=@path_to_your_image.jpg"
```

### Response

The API will return a JSON response containing the prediction results.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.