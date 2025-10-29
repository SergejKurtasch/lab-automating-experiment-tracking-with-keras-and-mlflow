'''Quick test script to verify MLflow integration works correctly
'''
import mlflow
from reuters_mlflow import train_model

def quick_test():
    """Run a quick test with minimal parameters"""
    print("Running quick test of MLflow integration...")
    
    # Set a test experiment
    mlflow.set_experiment("reuters_test")
    
    with mlflow.start_run(run_name="quick_test"):
        mlflow.set_tag("test_run", "true")
        
        # Run with minimal parameters for quick testing
        model, history, test_loss, test_accuracy = train_model(
            max_words=300,  # Very small vocabulary for speed
            batch_size=128, # Larger batch for speed
            epochs=1,       # Just 1 epoch for quick test
            learning_rate=0.001
        )
        
        print(f"Quick test completed!")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        return test_accuracy > 0.5  # Basic sanity check

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n✅ Quick test passed! MLflow integration is working correctly.")
        print("You can now run the full experiments:")
        print("  python reuters_mlflow.py")
        print("  python run_experiments.py")
    else:
        print("\n❌ Quick test failed. Please check the setup.")
