import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
from src.model import MnistCNN
from src.utils import count_parameters

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MnistCNN()
        print("\n" + "="*50)
        print("Running Model Architecture Tests...")
        print("="*50)

    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        self.assertLess(param_count, 25000, f"Model has {param_count} parameters, should be less than 25000")
        print(f"✓ Parameter Count Test Passed: Model has {param_count} parameters (< 25000)")

    def test_input_shape(self):
        batch_size = 1
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (batch_size, 10), "Output shape should be [batch_size, 10]")
        print("✓ Input Shape Test Passed: Model accepts 28x28 input correctly")

    def test_model_structure(self):
        # Test if model has expected number of layers
        num_conv_layers = sum(1 for m in self.model.modules() if isinstance(m, torch.nn.Conv2d))
        self.assertEqual(num_conv_layers, 3, "Model should have 3 convolutional layers")
        print(f"✓ Model Structure Test Passed: Found {num_conv_layers} convolutional layers")

if __name__ == '__main__':
    unittest.main() 