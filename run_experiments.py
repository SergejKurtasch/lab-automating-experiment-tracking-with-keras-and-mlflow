'''Script to run multiple experiments with different hyperparameters
for Reuters classification with MLflow tracking.
'''
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
    results = run_experiments()
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print("To view results in MLflow UI, run: mlflow ui")
    print("Then open: http://localhost:5000")
    print(f"{'='*60}")
