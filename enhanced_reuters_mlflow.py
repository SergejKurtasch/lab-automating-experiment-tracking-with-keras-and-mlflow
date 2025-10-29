"""
Enhanced Reuters Classification with MLflow Tracking

This script provides an enhanced version with comprehensive MLflow tracking,
experiment comparison, and better logging patterns. Recommended for advanced usage.

USAGE:
    1. Make sure MLflow is installed: pip install mlflow tensorflow keras
    2. Run this script: python enhanced_reuters_mlflow.py
    3. Start MLflow UI: mlflow ui
    4. Open browser: http://localhost:5000

WHAT IT DOES:
    - Trains two experiments (baseline and optimized configurations)
    - Compares experiment results automatically
    - Logs models, comprehensive metrics, and detailed parameters
    - Provides comparison tables and analysis

DIFFERENCES FROM reuters_mlflow.py:
    - More detailed parameter logging
    - Experiment comparison functionality
    - Model artifact logging
    - Best metrics tracking
    - Better organized experiment structure

SEE ALSO:
    - reuters_mlflow.py: Basic version (simpler, recommended to start here)
    - run_experiments.py: Run multiple experiments with different hyperparameters
"""
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
import mlflow.keras
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')

# Connect to existing MLflow UI using the same backend store (SQLite)
# This avoids 403 errors when connecting through REST API
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow_db_path = os.path.join(parent_dir, "mlflow.db")

if os.path.exists(mlflow_db_path):
    tracking_uri = f"sqlite:///{os.path.abspath(mlflow_db_path)}"
else:
    tracking_uri = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(tracking_uri)
print(f"✅ Connected to MLflow backend store: {mlflow.get_tracking_uri()}")

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

def load_and_preprocess_data(max_words=500, test_split=0.2):
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

def build_model(max_words, num_classes, hidden_units=512, dropout_rate=0.5):
    """Build the MLP model"""
    print('Building model...')
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def train_experiment(experiment_name, run_name, max_words=500, batch_size=64, 
                    epochs=2, learning_rate=0.001, hidden_units=512, dropout_rate=0.5):
    """Train model with comprehensive MLflow tracking"""
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        # Set tags for experiment identification
        mlflow.set_tag("project", "reuters_classification")
        mlflow.set_tag("model_type", "MLP")
        mlflow.set_tag("experiment_type", "hyperparameter_tuning")
        mlflow.set_tag("dataset", "reuters")
        mlflow.set_tag("device", device_name)
        
        # Load and preprocess data
        (x_train, y_train), (x_test, y_test), num_classes = load_and_preprocess_data(max_words)
        
        # Log all parameters comprehensively
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("max_words", max_words)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("hidden_units", hidden_units)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss_function", "categorical_crossentropy")
        mlflow.log_param("device_type", device_type)
        
        # Build model
        model = build_model(max_words, num_classes, hidden_units, dropout_rate)
        
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
        
        # Log final metrics explicitly (following MLflow Complete Guide pattern)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Log additional metrics from training history
        mlflow.log_metric("final_train_loss", history.history['loss'][-1])
        mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
        
        # Log best metrics during training
        mlflow.log_metric("best_train_accuracy", max(history.history['accuracy']))
        mlflow.log_metric("best_val_accuracy", max(history.history['val_accuracy']))
        mlflow.log_metric("min_train_loss", min(history.history['loss']))
        mlflow.log_metric("min_val_loss", min(history.history['val_loss']))
        
        # Log model using Keras flavor (correct API for Keras/TensorFlow models)
        mlflow.keras.log_model(
            model=model,
            artifact_path="model"
        )
        
        # Store run_id for later use
        run_id = run.info.run_id
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED")
        print("="*60)
        print(f"Run ID: {run_id}")
        print(f"Run Name: {run_name}")
        print(f"\nFinal Metrics:")
        print(f"  - Test Accuracy:  {test_accuracy:.4f}")
        print(f"  - Test Loss:      {test_loss:.4f}")
        print(f"  - Best Val Acc:   {max(history.history['val_accuracy']):.4f}")
        print(f"  - Min Val Loss:  {min(history.history['val_loss']):.4f}")
        print("\n✓ Model logged to MLflow")
        print(f"\n👉 View this run in the UI: http://localhost:5000")
        
        return {
            'run_id': run_id,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'best_val_accuracy': max(history.history['val_accuracy']),
            'min_val_loss': min(history.history['val_loss']),
            'model': model,
            'history': history
        }

def compare_experiments(experiment1, experiment2):
    """Compare two experiments and log comparison metrics"""
    print("\n" + "="*60)
    print("EXPERIMENT COMPARISON")
    print("="*60)
    
    # Calculate improvements
    accuracy_improvement = ((experiment2['test_accuracy'] - experiment1['test_accuracy']) / 
                           experiment1['test_accuracy']) * 100
    loss_improvement = ((experiment1['test_loss'] - experiment2['test_loss']) / 
                      experiment1['test_loss']) * 100
    
    # Create comparison table
    comparison_data = {
        'Metric': ['Test Accuracy', 'Test Loss', 'Best Val Accuracy', 'Min Val Loss'],
        'Experiment 1': [
            f"{experiment1['test_accuracy']:.4f}",
            f"{experiment1['test_loss']:.4f}",
            f"{experiment1['best_val_accuracy']:.4f}",
            f"{experiment1['min_val_loss']:.4f}"
        ],
        'Experiment 2': [
            f"{experiment2['test_accuracy']:.4f}",
            f"{experiment2['test_loss']:.4f}",
            f"{experiment2['best_val_accuracy']:.4f}",
            f"{experiment2['min_val_loss']:.4f}"
        ],
        'Improvement': [
            f"{accuracy_improvement:+.2f}%",
            f"{loss_improvement:+.2f}%",
            f"{((experiment2['best_val_accuracy'] - experiment1['best_val_accuracy']) / experiment1['best_val_accuracy'] * 100):+.2f}%",
            f"{((experiment1['min_val_loss'] - experiment2['min_val_loss']) / experiment1['min_val_loss'] * 100):+.2f}%"
        ]
    }
    
    # Print comparison table
    print(f"{'Metric':<20} {'Experiment 1':<15} {'Experiment 2':<15} {'Improvement':<12}")
    print("-" * 65)
    for i in range(len(comparison_data['Metric'])):
        print(f"{comparison_data['Metric'][i]:<20} {comparison_data['Experiment 1'][i]:<15} "
              f"{comparison_data['Experiment 2'][i]:<15} {comparison_data['Improvement'][i]:<12}")
    
    # Determine winner
    if experiment2['test_accuracy'] > experiment1['test_accuracy']:
        print(f"\n🏆 Experiment 2 WINS!")
        print(f"   Better accuracy: {accuracy_improvement:+.2f}% improvement")
        winner = "Experiment 2"
    elif experiment1['test_accuracy'] > experiment2['test_accuracy']:
        print(f"\n🏆 Experiment 1 WINS!")
        print(f"   Better accuracy: {-accuracy_improvement:+.2f}% improvement")
        winner = "Experiment 1"
    else:
        print(f"\n🤝 TIE!")
        print(f"   Both experiments achieved similar performance")
        winner = "Tie"
    
    return {
        'accuracy_improvement': accuracy_improvement,
        'loss_improvement': loss_improvement,
        'winner': winner,
        'comparison_data': comparison_data
    }

if __name__ == "__main__":
    """
    Main execution block.
    
    This script runs two experiments (baseline and optimized) and compares them.
    """
    print("="*60)
    print("ENHANCED REUTERS CLASSIFICATION WITH MLFLOW")
    print("Based on MLflow Complete Guide patterns")
    print("="*60)
    print("\nThis will run two experiments and compare them:\n")
    
    # Initialize MLflow client
    client = MlflowClient()
    
    # Experiment 1: Baseline Configuration
    print("\n" + "="*60)
    print("EXPERIMENT 1: BASELINE CONFIGURATION")
    print("="*60)
    
    experiment1 = train_experiment(
        experiment_name="reuters-classification-comparison",
        run_name="baseline-config",
        max_words=500,
        batch_size=64,
        epochs=2,
        learning_rate=0.001,
        hidden_units=512,
        dropout_rate=0.5
    )
    
    # Experiment 2: Optimized Configuration
    print("\n" + "="*60)
    print("EXPERIMENT 2: OPTIMIZED CONFIGURATION")
    print("="*60)
    
    experiment2 = train_experiment(
        experiment_name="reuters-classification-comparison",
        run_name="optimized-config",
        max_words=800,        # Larger vocabulary
        batch_size=32,         # Smaller batch size
        epochs=3,              # More epochs
        learning_rate=0.0005, # Lower learning rate
        hidden_units=256,      # Smaller hidden layer
        dropout_rate=0.3       # Less dropout
    )
    
    # Compare experiments
    comparison = compare_experiments(experiment1, experiment2)
    
    print(f"\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Winner: {comparison['winner']}")
    print(f"Accuracy improvement: {comparison['accuracy_improvement']:+.2f}%")
    print(f"Loss improvement: {comparison['loss_improvement']:+.2f}%")
    print(f"\n👉 View all experiments in MLflow UI: http://localhost:5000")
    print(f"🧪 View experiment: http://localhost:5000/#/experiments/1")
    print(f"🏃 View Experiment 1: http://localhost:5000/#/experiments/1/runs/{experiment1['run_id']}")
    print(f"🏃 View Experiment 2: http://localhost:5000/#/experiments/1/runs/{experiment2['run_id']}")
    
    print(f"\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n👉 Next steps:")
    print(f"   1. Start MLflow UI: mlflow ui")
    print(f"   2. Open browser: http://localhost:5000")
    print(f"   3. Look for experiment: 'reuters-classification-comparison'")
    print("="*60)
