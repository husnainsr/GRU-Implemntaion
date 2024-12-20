# GRU Implementation for DVEC Classification

## Overview
This project implements a Gated Recurrent Unit (GRU) neural network for classifying d-vector embeddings. The model is designed to distinguish between real and fake audio samples using their d-vector representations.

## Features
- Custom GRU architecture optimized for d-vector processing
- Comprehensive data preprocessing pipeline
- Early stopping mechanism to prevent overfitting
- Detailed logging system for training monitoring
- Training visualization with loss and accuracy plots
- Model checkpointing for best performance
- Gradient clipping for training stability

## Project Structure
```
.
├── models/             # Saved model checkpoints
├── logs/              # Training logs
├── plots/             # Training visualization plots
├── metrics/           # Training metrics in JSON format
├── DVECs2/           # Dataset directory
│   ├── real/         # Real audio d-vectors
│   └── fake/         # Fake audio d-vectors
└── gru_code.py       # Main implementation file
```

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Logging

## Model Architecture
```
GRUModel(
  (dropout_input): Dropout(p=0.4)
  (gru): GRU(66, 64, batch_first=True)
  (dropout1): Dropout(p=0.4)
  (fc1): Linear(64, 64)
  (dropout2): Dropout(p=0.4)
  (fc2): Linear(64, 2)
  (log_softmax): LogSoftmax(dim=1)
)
```

## Training Features
- Batch size: 32
- Learning rate: 1e-2
- Weight decay: 1e-4
- Early stopping patience: 7 epochs
- Gradient clipping: 1.0
- Train/Val split: 80/20

## Usage
1. Prepare your dataset in the DVECs2 directory
2. Run the training script:
```
python gru_code.py
```

## Monitoring
- Training progress is logged in `logs/`
- Training plots are saved in `plots/`
- Metrics are saved in `metrics/`
- Best model checkpoints are saved in `models/`

## Results
The model achieves:
- Training accuracy: ~95%
- Validation accuracy: ~65%
- Early stopping typically triggers around 30-40 epochs

## Future Improvements
- Implement cross-validation
- Add data augmentation
- Experiment with bidirectional GRU
- Try different optimizers
- Add inference script

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any queries, please open an issue in the repository.