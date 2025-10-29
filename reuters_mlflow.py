'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task with MLflow tracking.
'''
from __future__ import print_function

import numpy as np
import os

# Configure TensorFlow/Keras for GPU acceleration on Apple Silicon (M3)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for Apple Silicon
os.environ['TF_METAL_DEVICE_PLACEMENT'] = '1'  # Enable Metal device placement

import tensorflow as tf
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
import mlflow
import mlflow.tensorflow

# Connect to existing MLflow UI using the same backend store (SQLite)
# This avoids 403 errors when connecting through REST API
# The mlflow.db is in the parent directory (week38-MLFlow)
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow_db_path = os.path.join(parent_dir, "mlflow.db")

if os.path.exists(mlflow_db_path):
    tracking_uri = f"sqlite:///{os.path.abspath(mlflow_db_path)}"
else:
    # Fallback: try to use relative path or default
    tracking_uri = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(tracking_uri)
print(f"✅ Connected to MLflow backend store: {mlflow.get_tracking_uri()}")
print(f"   Database: {mlflow_db_path if os.path.exists(mlflow_db_path) else 'using default location'}")

# GPU Configuration for MacBook M3 (Metal Performance Shaders)
def configure_gpu():
    """Configure GPU settings for Apple Silicon Mac"""
    # Check available devices
    physical_devices = tf.config.list_physical_devices()
    print("\n" + "="*60)
    print("GPU/DEVICE CONFIGURATION")
    print("="*60)
    print(f"Available physical devices: {len(physical_devices)}")
    for device in physical_devices:
        print(f"  - {device.name}: {device.device_type}")
    
    # Check for Metal GPU (MPS) on Apple Silicon
    gpus = tf.config.list_physical_devices('GPU')
    cpu_devices = tf.config.list_physical_devices('CPU')
    
    if gpus:
        print("\n✅ GPU acceleration available (Metal Performance Shaders)")
        try:
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
            print(f"✅ Using GPU device: {gpus[0].name}")
            device_type = "GPU (Metal)"
            device_name = "Apple Silicon GPU (M3)"
        except Exception as e:
            print(f"⚠️  Warning: Could not configure GPU: {e}")
            device_type = "CPU"
            device_name = "CPU"
    else:
        print("\n⚠️  GPU not detected, using CPU")
        print("💡 Tip: Install tensorflow-metal for GPU acceleration:")
        print("   pip install tensorflow-metal")
        device_type = "CPU"
        device_name = "CPU"
    
    # Log TensorFlow version for debugging
    print(f"TensorFlow version: {tf.__version__}")
    print("="*60 + "\n")
    return device_type, device_name

# Configure GPU at module import
device_type, device_name = configure_gpu()

def load_and_preprocess_data(max_words=1000, test_split=0.2):
    """Load and preprocess Reuters dataset"""
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                             test_split=test_split)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    num_classes = np.max(y_train) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Convert class vector to binary class matrix '
          '(for use with categorical_crossentropy)')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    
    return (x_train, y_train), (x_test, y_test), num_classes

def build_model(max_words, num_classes):
    """Build the MLP model"""
    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def train_model(max_words=500, batch_size=64, epochs=2, learning_rate=0.001, run_name=None, use_existing_run=False):
    """Train the model with MLflow tracking
    
    Args:
        max_words: Maximum words in vocabulary
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        run_name: Optional name for the MLflow run (only used if creating new run)
        use_existing_run: If True, uses existing active MLflow run instead of creating new one
    """
    
    # Set MLflow experiment
    mlflow.set_experiment("reuters_classification")
    
    # Check if there's already an active run and we should use it
    active_run = mlflow.active_run()
    if use_existing_run and active_run is not None:
        # Use existing run (don't create new one)
        return _train_model_internal(max_words, batch_size, epochs, learning_rate)
    else:
        # Create new run
        with mlflow.start_run(run_name=run_name):
            return _train_model_internal(max_words, batch_size, epochs, learning_rate)

def _train_model_internal(max_words=500, batch_size=64, epochs=2, learning_rate=0.001):
    """Internal function to train model (assumes MLflow run is already active)"""
    # Set tag for experiment identification
    mlflow.set_tag("project", "reuters_classification")
    mlflow.set_tag("model_type", "MLP")
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test), num_classes = load_and_preprocess_data(max_words)
    
    # Log parameters
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("max_words", max_words)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("device_type", device_type)
    mlflow.set_tag("device", device_name)
    
    # Build model
    model = build_model(max_words, num_classes)
    
    # Enable MLflow autologging for TensorFlow/Keras
    mlflow.tensorflow.autolog()
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_split=0.1)
    
    # Evaluate model
    score = model.evaluate(x_test, y_test,
                         batch_size=batch_size, verbose=1)
    
    test_loss = score[0]
    test_accuracy = score[1]
    
    print('Test score:', test_loss)
    print('Test accuracy:', test_accuracy)
    
    # Log final metrics explicitly
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Log additional metrics from training history
    mlflow.log_metric("final_train_loss", history.history['loss'][-1])
    mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
    mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
    
    return model, history, test_loss, test_accuracy

if __name__ == "__main__":
    # Default parameters (optimized for speed)
    max_words = 500
    batch_size = 64
    epochs = 2
    learning_rate = 0.001
    
    print("Starting Reuters classification experiment with MLflow tracking...")
    model, history, test_loss, test_accuracy = train_model(
        max_words=max_words,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    print(f"Experiment completed! Test accuracy: {test_accuracy:.4f}")
    print(f"\n✅ Experiment saved to MLflow!")
    print(f"👉 Check your MLflow UI at http://localhost:5000")
    print(f"   Look for experiment: 'reuters_classification'")
