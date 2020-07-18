# Income-Distribution-ML

This program implements several machine learning and deep learning algorithms for prediction income distribution in California based on satellite imagery.

## Installation

The project requires the following frameworks:

- PyTorch: https://pytorch.org

- Scikit-learn: https://scikit-learn.org/stable/

- NumPy: https://numpy.org

- Pandas: https://pandas.pydata.org

## Usage

To run the program, use the following command:

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

## Authors

* **Tai Vu** - Stanford University
