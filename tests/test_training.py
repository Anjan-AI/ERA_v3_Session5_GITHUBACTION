import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
from src.train import train_model
from src.model import MnistCNN

class TestTraining(unittest.TestCase):
    def test_training_accuracy(self):
        model, accuracy, _ = train_model(epochs=1, batch_size=8)
        self.assertGreater(accuracy, 95.0, f"Training accuracy {accuracy:.2f}% is less than required 95%")

    def test_model_saving(self):
        _, _, model_path = train_model(epochs=1, batch_size=8)
        self.assertTrue(model_path.endswith('.pth'), "Model file should have .pth extension")

if __name__ == '__main__':
    unittest.main() 