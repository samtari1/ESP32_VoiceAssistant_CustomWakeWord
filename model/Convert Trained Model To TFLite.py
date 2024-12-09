import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("trained.keras")

# Convert the loaded model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format.")
