"""
Model utilities for garbage detection
"""
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

# 12 Kategori Label Sampah
LABELS = [
    'battery',
    'biological', 
    'brown-glass',
    'cardboard',
    'clothes',
    'green-glass',
    'metal',
    'paper',
    'plastic',
    'shoes',
    'trash',
    'white-glass'
]

LABEL_NAMES = {
    'battery': 'Baterai',
    'biological': 'Sampah Organik/Biologis',
    'brown-glass': 'Kaca Coklat',
    'cardboard': 'Kardus',
    'clothes': 'Pakaian Bekas',
    'green-glass': 'Kaca Hijau',
    'metal': 'Logam/Metal',
    'paper': 'Kertas',
    'plastic': 'Plastik',
    'shoes': 'Sepatu Bekas',
    'trash': 'Sampah Umum',
    'white-glass': 'Kaca Putih/Bening'
}

RECYCLABLE_INFO = {
    'battery': {'recyclable': False, 'category': 'Berbahaya', 'disposal': 'Tempat khusus baterai bekas'},
    'biological': {'recyclable': True, 'category': 'Organik', 'disposal': 'Kompos atau tempat sampah organik'},
    'brown-glass': {'recyclable': True, 'category': 'Kaca', 'disposal': 'Tempat sampah kaca coklat'},
    'cardboard': {'recyclable': True, 'category': 'Kertas', 'disposal': 'Tempat sampah kertas/kardus'},
    'clothes': {'recyclable': True, 'category': 'Tekstil', 'disposal': 'Donasi atau daur ulang tekstil'},
    'green-glass': {'recyclable': True, 'category': 'Kaca', 'disposal': 'Tempat sampah kaca hijau'},
    'metal': {'recyclable': True, 'category': 'Logam', 'disposal': 'Tempat sampah logam'},
    'paper': {'recyclable': True, 'category': 'Kertas', 'disposal': 'Tempat sampah kertas'},
    'plastic': {'recyclable': True, 'category': 'Plastik', 'disposal': 'Tempat sampah plastik'},
    'shoes': {'recyclable': False, 'category': 'Campuran', 'disposal': 'Tempat sampah umum atau donasi'},
    'trash': {'recyclable': False, 'category': 'Umum', 'disposal': 'Tempat sampah umum'},
    'white-glass': {'recyclable': True, 'category': 'Kaca', 'disposal': 'Tempat sampah kaca putih/bening'}
}

def load_model(model_path):
    """
    Load trained model
    
    Args:
        model_path: path to the model file
    
    Returns:
        model: loaded TensorFlow model or None if failed
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Validate model output shape
        expected_outputs = len(LABELS)
        if hasattr(model, 'output_shape'):
            actual_outputs = model.output_shape[-1]
            if actual_outputs != expected_outputs:
                logger.warning(f"Model output shape mismatch. Expected: {expected_outputs}, Got: {actual_outputs}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def predict_with_model(model, processed_image, confidence_threshold=0.05):
    """
    Make prediction using the loaded model
    
    Args:
        model: loaded TensorFlow model
        processed_image: preprocessed image array
        confidence_threshold: minimum confidence threshold
    
    Returns:
        list: prediction results
    """
    try:
        # Make prediction
        predictions = model.predict(processed_image)[0]
        
        # Create results
        results = []
        for i, (label, confidence) in enumerate(zip(LABELS, predictions)):
            if confidence > confidence_threshold:
                recyclable_info = RECYCLABLE_INFO[label]
                results.append({
                    'class': label,
                    'class_name': LABEL_NAMES[label],
                    'confidence': float(confidence),
                    'recyclable': recyclable_info['recyclable'],
                    'category': recyclable_info['category'],
                    'disposal_method': recyclable_info['disposal']
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results[:3]  # Return top 3
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None

def mock_prediction(confidence_threshold=0.05):
    """
    Generate mock prediction for testing
    
    Args:
        confidence_threshold: minimum confidence threshold
    
    Returns:
        list: mock prediction results
    """
    import random
    
    # Generate random predictions
    predictions = np.random.rand(len(LABELS))
    
    # Make some predictions higher for realism
    top_indices = np.random.choice(len(LABELS), size=2, replace=False)
    for idx in top_indices:
        predictions[idx] = np.random.uniform(0.7, 0.95)
    
    # Normalize
    predictions = predictions / np.sum(predictions)
    
    # Create results
    results = []
    for i, (label, confidence) in enumerate(zip(LABELS, predictions)):
        if confidence > confidence_threshold:
            recyclable_info = RECYCLABLE_INFO[label]
            results.append({
                'class': label,
                'class_name': LABEL_NAMES[label],
                'confidence': float(confidence),
                'recyclable': recyclable_info['recyclable'],
                'category': recyclable_info['category'],
                'disposal_method': recyclable_info['disposal']
            })
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return results[:3]  # Return top 3

def get_model_info():
    """
    Get information about the model and labels
    
    Returns:
        dict: model information
    """
    return {
        'total_labels': len(LABELS),
        'labels': LABELS,
        'label_names': LABEL_NAMES,
        'recyclable_count': sum(1 for info in RECYCLABLE_INFO.values() if info['recyclable']),
        'non_recyclable_count': sum(1 for info in RECYCLABLE_INFO.values() if not info['recyclable'])
    }