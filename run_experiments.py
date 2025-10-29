"""
Run Multiple Experiments with Different Hyperparameters

This script runs multiple experiments with different hyperparameter combinations
to find the best configuration. It uses the basic train_model() function from
reuters_mlflow.py and runs it with various parameter settings.

USAGE:
    1. Make sure you have run reuters_mlflow.py at least once (to ensure dependencies)
    2. Install dependencies: pip install mlflow tensorflow keras
    3. Run this script: python run_experiments.py
    4. Start MLflow UI: mlflow ui
    5. Open browser: http://localhost:5000

WHAT IT DOES:
    - Runs 7 different experiments with varying hyperparameters:
      * Baseline configuration
      * Larger/smaller vocabulary
      * Different batch sizes
      * More epochs
      * Different learning rates
    - Compares all experiments and shows results ranked by test accuracy
    - All results are logged to MLflow for easy comparison

EXPERIMENT CONFIGURATIONS:
    - baseline: Standard configuration
    - larger_vocabulary: More words in vocabulary
    - smaller_batch: Smaller batch size
    - larger_batch: Larger batch size
    - more_epochs: More training epochs
    - higher_lr: Higher learning rate
    - lower_lr: Lower learning rate

TO MODIFY EXPERIMENTS:
    Edit the 'experiments' list in the run_experiments() function.

SEE ALSO:
    - reuters_mlflow.py: Basic single experiment script
    - enhanced_reuters_mlflow.py: Enhanced version with comparison features
"""
import mlflow

# Connect to existing MLflow UI using the same backend store (SQLite)
# (reuters_mlflow will also set this, but we set it here for clarity)
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow_db_path = os.path.join(parent_dir, "mlflow.db")

if os.path.exists(mlflow_db_path):
    tracking_uri = f"sqlite:///{os.path.abspath(mlflow_db_path)}"
else:
    tracking_uri = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(tracking_uri)

from reuters_mlflow import train_model

def run_experiments():
    """Run multiple experiments with different hyperparameters"""
    
    # Define different parameter combinations to test (optimized for speed)
    experiments = [
        {
            "name": "baseline",
            "max_words": 500,
            "batch_size": 64,
            "epochs": 2,
            "learning_rate": 0.001
        },
        {
            "name": "larger_vocabulary",
            "max_words": 800,
            "batch_size": 64,
            "epochs": 2,
            "learning_rate": 0.001
        },
        {
            "name": "smaller_batch",
            "max_words": 500,
            "batch_size": 32,
            "epochs": 2,
            "learning_rate": 0.001
        },
        {
            "name": "larger_batch",
            "max_words": 500,
            "batch_size": 128,
            "epochs": 2,
            "learning_rate": 0.001
        },
        {
            "name": "more_epochs",
            "max_words": 500,
            "batch_size": 64,
            "epochs": 3,
            "learning_rate": 0.001
        },
        {
            "name": "higher_lr",
            "max_words": 500,
            "batch_size": 64,
            "epochs": 2,
            "learning_rate": 0.01
        },
        {
            "name": "lower_lr",
            "max_words": 500,
            "batch_size": 64,
            "epochs": 2,
            "learning_rate": 0.0001
        }
    ]
    
    print("Starting multiple experiments with different hyperparameters...")
    print(f"Total experiments to run: {len(experiments)}")
    
    results = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'='*50}")
        print(f"Running experiment {i}/{len(experiments)}: {exp_config['name']}")
        print(f"Parameters: {exp_config}")
        print(f"{'='*50}")
        
        try:
            # Set run name for better identification in MLflow UI
            mlflow.set_experiment("reuters_classification")
            
            # Create run with name, then call train_model which will use nested=False
            # This allows train_model to use the existing run context
            with mlflow.start_run(run_name=exp_config['name']):
                # Add experiment name as tag
                mlflow.set_tag("experiment_name", exp_config['name'])
                
                # Train model with current parameters
                # Pass use_existing_run=True so train_model uses the run we just created
                model, history, test_loss, test_accuracy = train_model(
                    max_words=exp_config['max_words'],
                    batch_size=exp_config['batch_size'],
                    epochs=exp_config['epochs'],
                    learning_rate=exp_config['learning_rate'],
                    use_existing_run=True  # Use the run we created above
                )
                
                # Store results
                result = {
                    'name': exp_config['name'],
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'config': exp_config
                }
                results.append(result)
                
                print(f"Experiment '{exp_config['name']}' completed!")
                print(f"Test accuracy: {test_accuracy:.4f}")
                print(f"Test loss: {test_loss:.4f}")
                
        except Exception as e:
            print(f"Error in experiment '{exp_config['name']}': {str(e)}")
            continue
    
    # Print summary of all experiments
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    # Sort by test accuracy (descending)
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    print(f"{'Rank':<5} {'Experiment':<20} {'Test Accuracy':<15} {'Test Loss':<12}")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<5} {result['name']:<20} {result['test_accuracy']:<15.4f} {result['test_loss']:<12.4f}")
    
    print(f"\nBest performing experiment: {results[0]['name']}")
    print(f"Best test accuracy: {results[0]['test_accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    """
    Main execution block.
    
    This will run all experiments defined in run_experiments() function.
    """
    print("="*60)
    print("RUNNING MULTIPLE EXPERIMENTS")
    print("="*60)
    print("\nThis script will run 7 different experiments with various hyperparameters.")
    print("This may take a while depending on your hardware...\n")
    
    results = run_experiments()
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print(f"\n✅ Total experiments run: {len(results)}")
    print(f"📊 Best accuracy: {max(r['test_accuracy'] for r in results):.4f}")
    print(f"\n👉 Next steps:")
    print(f"   1. Start MLflow UI: mlflow ui")
    print(f"   2. Open browser: http://localhost:5000")
    print(f"   3. Look for experiment: 'reuters_classification'")
    print(f"   4. Compare experiments using the MLflow UI interface")
    print("="*60)
