import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
from src.train import train_model
from src.model import MnistCNN

class TestTraining(unittest.TestCase):
    def setUp(self):
        print("\n" + "="*50)
        print("Running Training Pipeline Tests...")
        print("="*50)

    def test_training_accuracy(self):
        print("\nTraining model for accuracy test...")
        model, accuracy, _ = train_model(epochs=1, batch_size=8)
        self.assertGreater(accuracy, 95.0, f"Training accuracy {accuracy:.2f}% is less than required 95%")
        print(f"✓ Training Accuracy Test Passed: Achieved {accuracy:.2f}% (> 95%)")

    def test_model_saving(self):
        print("\nTesting model saving functionality...")
        _, _, model_path = train_model(epochs=1, batch_size=8)
        self.assertTrue(model_path.endswith('.pth'), "Model file should have .pth extension")
        print(f"✓ Model Saving Test Passed: Model saved as {model_path}")

    def tearDown(self):
        print("-"*50)

if __name__ == '__main__':
    unittest.main(verbosity=2) 