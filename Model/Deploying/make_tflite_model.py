'''
Python script to convert saved keras model to tf lite for deployment on small-devices
'''
# Import libraries
import tensorflow as tf
import json
# Load model
model = tf.keras.models.load_model('model.keras')

# Initialize converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Quantization for size and latency reduction (This could reduce some accuracy though)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Add metadata
with open('model_metadata.json') as f:
    metadata = json.load(f)

converter.metadata = metadata

# Convert to tflite format
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print('Model converted and saved successfully!')

