'''Quick test for enhanced MLflow integration
Tests the improved logging patterns from MLflow Complete Guide
'''
import mlflow
from enhanced_reuters_mlflow import train_experiment, compare_experiments

def quick_test():
    """Run a quick test with minimal parameters"""
    print("Running quick test of enhanced MLflow integration...")
    
    # Set a test experiment
    mlflow.set_experiment("reuters-quick-test")
    
    # Test Experiment 1: Minimal configuration
    print("\n" + "="*50)
    print("QUICK TEST - EXPERIMENT 1")
    print("="*50)
    
    experiment1 = train_experiment(
        experiment_name="reuters-quick-test",
        run_name="quick-test-1",
        max_words=200,  # Very small vocabulary
        batch_size=128, # Large batch for speed
        epochs=1,       # Just 1 epoch
        learning_rate=0.01,  # Higher LR for faster convergence
        hidden_units=64,     # Small model
        dropout_rate=0.3
    )
    
    # Test Experiment 2: Different configuration
    print("\n" + "="*50)
    print("QUICK TEST - EXPERIMENT 2")
    print("="*50)
    
    experiment2 = train_experiment(
        experiment_name="reuters-quick-test",
        run_name="quick-test-2",
        max_words=300,  # Slightly larger vocabulary
        batch_size=64,  # Smaller batch
        epochs=1,       # Just 1 epoch
        learning_rate=0.005,  # Lower LR
        hidden_units=128,      # Larger model
        dropout_rate=0.5
    )
    
    # Quick comparison
    comparison = compare_experiments(experiment1, experiment2)
    
    print(f"\n" + "="*50)
    print("QUICK TEST RESULTS")
    print("="*50)
    print(f"✅ Both experiments completed successfully!")
    print(f"🏆 Winner: {comparison['winner']}")
    print(f"📊 Accuracy difference: {comparison['accuracy_improvement']:+.2f}%")
    print(f"🔗 View in MLflow UI: http://localhost:5000")
    
    return experiment1['test_accuracy'] > 0.3 and experiment2['test_accuracy'] > 0.3

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n✅ Quick test PASSED! Enhanced MLflow integration is working correctly.")
        print("\nYou can now run the full experiments:")
        print("  python enhanced_reuters_mlflow.py")
        print("  python comprehensive_experiments.py")
    else:
        print("\n❌ Quick test FAILED. Please check the setup.")
