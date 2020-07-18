# Income-Distribution-ML

This program implements several machine learning models for mapping income distribution using satellite imagery.

## Usage

```bash
python main.py
```

There are several optional command line arguments:

- --model: Machine Learning algorithm, including 'lr', 'rf', 'knn', 'svm', and 'cnn'.
- --arch: ResNet architecture, such as 'resnet18' or 'resnet34'.
- --regularize: Regularization techniques, such as 'ridge' or 'lasso'.
- --batch-size: Size of a training batch for CNN.
- --lr: Learning rate for CNN.
- --epochs: Number of training epochs for CNN.
