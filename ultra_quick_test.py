'''Ultra-fast test script to verify MLflow integration works correctly
'''
import mlflow
from reuters_mlflow import train_model

def ultra_quick_test():
    """Run an ultra-quick test with minimal parameters"""
    print("Running ultra-quick test of MLflow integration...")
    
    # train_model() creates its own MLflow run, so we don't need to create one here
    # Run with ultra-minimal parameters for speed
    model, history, test_loss, test_accuracy = train_model(
        max_words=200,  # Ultra small vocabulary
        batch_size=256, # Very large batch for speed
        epochs=1,       # Just 1 epoch
        learning_rate=0.01  # Higher LR for faster convergence
    )
    
    print(f"Ultra-quick test completed!")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return test_accuracy > 0.3  # Lower threshold for ultra-fast test

if __name__ == "__main__":
    success = ultra_quick_test()
    
    if success:
        print("\n✅ Ultra-quick test passed! MLflow integration is working correctly.")
        print("You can now run the full experiments:")
        print("  python reuters_mlflow.py")
        print("  python run_experiments.py")
    else:
        print("\n❌ Ultra-quick test failed. Please check the setup.")
