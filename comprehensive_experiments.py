'''Advanced Experiment Comparison Script
Demonstrates comprehensive MLflow tracking with detailed analysis and visualization.
'''
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

# Connect to existing MLflow UI using the same backend store (SQLite)
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow_db_path = os.path.join(parent_dir, "mlflow.db")

if os.path.exists(mlflow_db_path):
    tracking_uri = f"sqlite:///{os.path.abspath(mlflow_db_path)}"
else:
    tracking_uri = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(tracking_uri)

from enhanced_reuters_mlflow import train_experiment, compare_experiments

def run_comprehensive_experiments():
    """Run multiple experiments with different configurations for comprehensive comparison"""
    
    print("="*80)
    print("COMPREHENSIVE REUTERS CLASSIFICATION EXPERIMENTS")
    print("Based on MLflow Complete Guide - Advanced Tracking Patterns")
    print("="*80)
    
    # Initialize MLflow client
    client = MlflowClient()
    
    # Define experiment configurations
    experiments_config = [
        {
            "name": "baseline",
            "description": "Standard configuration with default parameters",
            "params": {
                "max_words": 500,
                "batch_size": 64,
                "epochs": 2,
                "learning_rate": 0.001,
                "hidden_units": 512,
                "dropout_rate": 0.5
            }
        },
        {
            "name": "optimized",
            "description": "Optimized configuration with tuned hyperparameters",
            "params": {
                "max_words": 800,
                "batch_size": 32,
                "epochs": 3,
                "learning_rate": 0.0005,
                "hidden_units": 256,
                "dropout_rate": 0.3
            }
        },
        {
            "name": "conservative",
            "description": "Conservative approach with smaller model and lower learning rate",
            "params": {
                "max_words": 300,
                "batch_size": 128,
                "epochs": 2,
                "learning_rate": 0.0001,
                "hidden_units": 128,
                "dropout_rate": 0.7
            }
        },
        {
            "name": "aggressive",
            "description": "Aggressive approach with larger model and higher learning rate",
            "params": {
                "max_words": 1000,
                "batch_size": 16,
                "epochs": 4,
                "learning_rate": 0.01,
                "hidden_units": 1024,
                "dropout_rate": 0.2
            }
        }
    ]
    
    # Run all experiments
    experiment_results = []
    
    for i, config in enumerate(experiments_config, 1):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i}: {config['name'].upper()}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        result = train_experiment(
            experiment_name="reuters-comprehensive-comparison",
            run_name=f"{config['name']}-experiment",
            **config['params']
        )
        
        # Add configuration info to result
        result['config_name'] = config['name']
        result['config_description'] = config['description']
        result['config_params'] = config['params']
        
        experiment_results.append(result)
    
    # Comprehensive comparison
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    
    # Create detailed comparison table
    comparison_data = []
    for result in experiment_results:
        comparison_data.append({
            'Experiment': result['config_name'],
            'Description': result['config_description'],
            'Test Accuracy': result['test_accuracy'],
            'Test Loss': result['test_loss'],
            'Best Val Accuracy': result['best_val_accuracy'],
            'Min Val Loss': result['min_val_loss'],
            'Max Words': result['config_params']['max_words'],
            'Batch Size': result['config_params']['batch_size'],
            'Epochs': result['config_params']['epochs'],
            'Learning Rate': result['config_params']['learning_rate'],
            'Hidden Units': result['config_params']['hidden_units'],
            'Dropout Rate': result['config_params']['dropout_rate']
        })
    
    # Convert to DataFrame for better formatting
    df = pd.DataFrame(comparison_data)
    
    print("\nDETAILED RESULTS TABLE:")
    print("-" * 120)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Find best performing experiment
    best_accuracy_idx = df['Test Accuracy'].idxmax()
    best_loss_idx = df['Test Loss'].idxmin()
    
    print(f"\n🏆 BEST PERFORMING EXPERIMENTS:")
    print(f"   Best Accuracy: {df.iloc[best_accuracy_idx]['Experiment']} ({df.iloc[best_accuracy_idx]['Test Accuracy']:.4f})")
    print(f"   Best Loss:      {df.iloc[best_loss_idx]['Experiment']} ({df.iloc[best_loss_idx]['Test Loss']:.4f})")
    
    # Calculate improvements over baseline
    baseline_idx = df[df['Experiment'] == 'baseline'].index[0]
    baseline_accuracy = df.iloc[baseline_idx]['Test Accuracy']
    baseline_loss = df.iloc[baseline_idx]['Test Loss']
    
    print(f"\n📊 IMPROVEMENTS OVER BASELINE:")
    print("-" * 50)
    for idx, row in df.iterrows():
        if idx != baseline_idx:
            acc_improvement = ((row['Test Accuracy'] - baseline_accuracy) / baseline_accuracy) * 100
            loss_improvement = ((baseline_loss - row['Test Loss']) / baseline_loss) * 100
            print(f"   {row['Experiment']:<12}: Accuracy {acc_improvement:+.2f}%, Loss {loss_improvement:+.2f}%")
    
    # Parameter analysis
    print(f"\n🔍 PARAMETER ANALYSIS:")
    print("-" * 50)
    
    # Analyze correlation between parameters and performance
    param_analysis = {
        'Max Words': df['Max Words'].corr(df['Test Accuracy']),
        'Batch Size': df['Batch Size'].corr(df['Test Accuracy']),
        'Epochs': df['Epochs'].corr(df['Test Accuracy']),
        'Learning Rate': df['Learning Rate'].corr(df['Test Accuracy']),
        'Hidden Units': df['Hidden Units'].corr(df['Test Accuracy']),
        'Dropout Rate': df['Dropout Rate'].corr(df['Test Accuracy'])
    }
    
    print("Correlation with Test Accuracy:")
    for param, corr in param_analysis.items():
        direction = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.5 else "weak"
        print(f"   {param:<15}: {corr:+.3f} ({strength} {direction})")
    
    # Generate MLflow UI links
    print(f"\n🔗 MLFLOW UI LINKS:")
    print("-" * 50)
    print(f"   Main UI: http://localhost:5000")
    print(f"   Experiment: http://localhost:5000/#/experiments/1")
    
    for result in experiment_results:
        print(f"   {result['config_name']:<12}: http://localhost:5000/#/experiments/1/runs/{result['run_id']}")
    
    return experiment_results, df

def analyze_hyperparameter_impact(experiment_results):
    """Analyze the impact of different hyperparameters on model performance"""
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER IMPACT ANALYSIS")
    print(f"{'='*80}")
    
    # Group experiments by parameter changes
    analysis = {
        'Vocabulary Size Impact': [],
        'Batch Size Impact': [],
        'Learning Rate Impact': [],
        'Model Size Impact': []
    }
    
    for result in experiment_results:
        config = result['config_params']
        
        # Vocabulary size analysis
        analysis['Vocabulary Size Impact'].append({
            'max_words': config['max_words'],
            'accuracy': result['test_accuracy']
        })
        
        # Batch size analysis
        analysis['Batch Size Impact'].append({
            'batch_size': config['batch_size'],
            'accuracy': result['test_accuracy']
        })
        
        # Learning rate analysis
        analysis['Learning Rate Impact'].append({
            'learning_rate': config['learning_rate'],
            'accuracy': result['test_accuracy']
        })
        
        # Model size analysis
        analysis['Model Size Impact'].append({
            'hidden_units': config['hidden_units'],
            'accuracy': result['test_accuracy']
        })
    
    # Print analysis
    for category, data in analysis.items():
        print(f"\n{category}:")
        print("-" * 40)
        for item in data:
            param_name = list(item.keys())[0]
            param_value = item[param_name]
            accuracy = item['accuracy']
            print(f"   {param_name}: {param_value:<8} → Accuracy: {accuracy:.4f}")
    
    return analysis

if __name__ == "__main__":
    # Run comprehensive experiments
    results, comparison_df = run_comprehensive_experiments()
    
    # Analyze hyperparameter impact
    impact_analysis = analyze_hyperparameter_impact(results)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Completed {len(results)} experiments")
    print(f"📊 Best accuracy: {comparison_df['Test Accuracy'].max():.4f}")
    print(f"📉 Best loss: {comparison_df['Test Loss'].min():.4f}")
    print(f"🔗 View all results in MLflow UI: http://localhost:5000")
    print(f"{'='*80}")
