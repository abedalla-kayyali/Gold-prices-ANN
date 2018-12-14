# Gold-prices-ANN

A generic ANN Regressor used for predicting gold prices in USD. Available [here](https://datahub.io/core/gold-prices#data).  
Made using the Tensorflow framework in python

## Usage

### Same dataset

Download the data and adjust `DATA_PATH` in `data_importer.py` accordingly.

### Other datasets

The `ANNRegressor` can be used for any regression problem. The parameter's can be passed in the constructor and `fit`/`k_fold_x_validation` method for training.  
Weights can be saved and loaded during training and testing respectively.  
Predictions can be made via the `predict` method.  
For further details please view the `ann.py` file, it contains docstrings and comments explaining each step of the process.

## Model properties

A sequential neural network with the following implementations.

- Batched gradient descent.
- Nestrov's momentum to improve learning with batches.
- RMSProp for adaptive learning rate.
- Dropout regularization.
- K-Fold cross validation.

## Model Parameters

For the prediction of gold prices.

- Two hidden layer with depths (15, 15)
- Learning rate (0.005)
- Layer's probability of keeping (1, 1, 1)
- RMSProp cache decay (0.9)
- Nestrov's momentum (0.9)
- Epochs (1,000)
- Test/validation split (10) of which 1 part is validation
- Batch size (30)

## Results

Output of training

```(bash)
Getting data...
Initializing model...
Training model...
100  iterations,  42.64995455741882 % R squared...
200  iterations,  61.57615780830383 % R squared...
300  iterations,  89.15794938802719 % R squared...
400  iterations,  85.2174773812294 % R squared...
500  iterations,  90.28166085481644 % R squared...
600  iterations,  79.8068642616272 % R squared...
700  iterations,  90.28975814580917 % R squared...
800  iterations,  86.97638064622879 % R squared...
900  iterations,  90.86299017071724 % R squared...
R squared for training:  93.05461645126343 %
R squared for testing:  92.79382154345512 %
```
