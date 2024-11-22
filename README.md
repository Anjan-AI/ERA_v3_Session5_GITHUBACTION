# MNIST CNN Model with CI/CD Pipeline

This project implements a CNN model for MNIST digit classification with a complete CI/CD pipeline.

## Project Structure 

project/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml
├── src/
│   ├── __init__.py    # Empty file
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── __init__.py    # Empty file
│   ├── test_model.py
│   └── test_training.py
├── setup.py           # New file
├── requirements.txt
└── README.md


## Requirements
-torch==2.0.1
-torchvision==0.15.2
numpy==1.24.3
-pytest==7.3.1
setuptools>=65.5.1

## Local Setup and Testing

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install package in development mode:
```bash
pip install -e .
```

3. Run tests locally:
```bash
python -m unittest discover tests/
or
python run_tests.py
```

4. Train model:
```bash
python src/train.py
```


## CI/CD Pipeline

The GitHub Actions workflow will:
1. Run all tests
2. Train the model
3. Save the trained model as an artifact

## Model Details
- Architecture: CNN with BatchNorm
- Parameters: <25,000
- Input shape: 28x28
- Output: 10 classes
- Target accuracy: >95% training accuracy in 1 epoch

## Deployment
Models are automatically saved with timestamps and accuracy metrics in the filename format:
`mnist_model_<accuracy>acc_<timestamp>.pth`

## GitHub Actions
The pipeline will run automatically on every push to the repository. You can view the results in the Actions tab.

## Notes
- The model is trained on CPU in GitHub Actions
- Trained models are saved as artifacts in the workflow
- All tests must pass for successful deployment


To use this project:
1. Clone the repository
2. Create a new branch for your changes
3. Push your changes to the new branch
4. Create a pull request to merge your changes into the main branch
5. The workflow will automatically run and deploy your changes if all tests pass