#------------------
# Load modules
#------------------
import pandas as pd
import numpy as np; np.random.seed(12345)
import tensorflow as tf
import datetime as dt
import time
import matplotlib.pyplot as plt
import sklearn

from tensorflow.keras import models
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.activations import *

from sklearn.metrics import  mean_squared_error, mean_absolute_error
#---------------
# User parameters
#---------------

k = 10;						# Number of folds for k-fold cross validation
num_epochs = 30;			# Number of epochs for training
features = 'FWI';			# Options are 'STFWI,' 'STM,' 'FWI,' 'M,' or 'ALL'

#-----------------------
# GridSearch parameters to be optimized
#-----------------------
#optimizers = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam','Ftrl']
neurons = [2,4,8,16,32,64]
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#activations = ['relu','tanh']
batches = [1]
#init = ['normal','uniform']

#------------------
# Load forest fire data
#------------------
ff_df = pd.read_csv("../Data/forestfires.csv") # Load as Pandas Dataframe

# Convert month str to number
ff_df['month'] = [dt.datetime.strptime(x, '%b').month for x in ff_df['month'] ]
# Convert day str to number. mon = 0, sun = 6
ff_df['day'] = [time.strptime(x, '%a').tm_wday for x in ff_df['day'] ]

ff = np.array(ff_df) # Convert to numpy array

area_mean = np.mean(ff[:,12])
area_std = np.std(ff[:,12])

print("Area mean:", area_mean)
print("Area std: ",area_std)

# ---------------------------------------------------------------------------- #
# Feature Selection
# ---------------------------------------------------------------------------- #

# Select the features based on user parameters
if features == 'STFWI':
    ff_features = ff[:,0:8]
    ff_response = ff[:,12]

if features == 'FWI':
    ff_features = ff[:,4:8]
    ff_response = ff[:,12]

if features == 'M':
    ff_features = ff[:,8:11]
    ff_response = ff[:,12]

if features == 'STM':
    ff_features = ff[:,[0,1,2,3,8,9,10,11]]
    ff_response = ff[:,12]

if features == 'ALL':
    ff_features = ff[:,0:11]
    ff_response = ff[:,12]  

#---------------
# Split into training and test data
#---------------

indices = np.random.permutation(ff.shape[0])
CUT = round(0.8*ff.shape[0])
train_idx, test_idx = indices[:CUT], indices[CUT:]
train_data, test_data = ff_features[train_idx,:], ff_features[test_idx,:]
train_targets, test_targets = ff_response[train_idx], ff_response[test_idx]

#------------------
# Examine data
#------------------

print("Shape of training data:",train_data.shape)
print("Shape of test data:",test_data.shape)

print("Shape of training targets:",train_targets.shape)
print("Shape of test targets:", test_targets.shape)

#------------------
# Normalize data
#------------------
# For X
Xmean = train_data.mean(axis=0)
train_data -= Xmean
Xstd = train_data.astype('float32').std(axis=0)
train_data /= Xstd

test_data -= Xmean
test_data /= Xstd

# For Y
Ymean = train_targets.mean(axis=0)
train_targets -= Ymean
Ystd = train_targets.astype('float32').std(axis=0)
train_targets /= Ystd

test_targets -= Ymean
test_targets /= Ystd

print("Mean of features:",Xmean)
print("Standard deviation of features:",Xstd)
print("Mean of burned area:",Ymean)
print("Standard deviation of burned area:",Ystd)

#------------------
# Define model architecture to be optimized
#------------------
def build_model(neurons=2,
                dropout_rates = 0.0):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dropout(dropout_rates))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                loss='mae')
    return model

model = KerasRegressor(build_fn=build_model, epochs=num_epochs,batch_size=1, verbose=1)

# ---------------------------------------------------------------------------- #
# Run grid search
# ---------------------------------------------------------------------------- #

param_grid = dict(neurons = neurons,
                  dropout_rates = dropout_rates)
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    scoring = 'neg_mean_absolute_error',
                    cv = k,
                    verbose = 0,
                    refit = True,
                    return_train_score=True,
                    n_jobs=-1)
grid_result = grid.fit(train_data, train_targets)

# ---------------------------------------------------------------------------- #
# Save results
# ---------------------------------------------------------------------------- #


# Run on test data

pred_targets = grid_result.predict(test_data)

print(f'Best params: {grid_result.best_params_}')
print(f'Best score: {grid_result.best_score_}')
print(f'Test MAE: {mean_absolute_error(y_true=test_targets, y_pred=pred_targets)}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_true=test_targets, y_pred=pred_targets))}')

print("Best MAE (unnormalized):", Ystd*mean_absolute_error(y_true=test_targets, y_pred=pred_targets))
print("Best RMSE (unnormalized):", Ystd*np.sqrt(mean_squared_error(y_true=test_targets, y_pred=pred_targets)))

# Other scores
# From train split
means_train = grid_result.cv_results_['mean_train_score']
# From test split
means_val = grid_result.cv_results_['mean_test_score']

params = grid_result.cv_results_['params']

print("\nOther scores:")

for param, mean_t, mean_v in zip(params, means_train, means_val):
    print("%r: train: %f, val: %f" % (param, mean_t, mean_v))
    


# ---------------------------------------------------------------------------- #    
# Plot results of grid search
# ---------------------------------------------------------------------------- #


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))
    # Plot Grid search scores
    _, ax = plt.subplots(1,1)
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
    ax.set_title("Grid Search Scores", fontsize=15, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=15)
    ax.set_ylabel('CV Average Score', fontsize=15)
    ax.legend(loc="best", fontsize=10)
    plt.savefig('../Plots/Grid Search (One Hidden Layer) Summary.png')
    plt.show()


plot_grid_search(grid.cv_results_,
                 neurons,
                 dropout_rates,
                 'Neurons in Hidden Layer',
                 'Dropout Rate')



# ---------------------------------------------------------------------------- #
# Fit the best model after grid search
# ---------------------------------------------------------------------------- #


# Get a fresh, compiled model.
best_model = build_model(neurons = grid_result.best_params_["neurons"],
                         dropout_rates = grid_result.best_params_["dropout_rates"],)
# Train it on the entirety of the data.
best_model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)

# Fit best_model on test data
test_data = np.asarray(test_data).astype(np.float32)
test_targets = np.asarray(test_targets).astype(np.float32)

test_mae_score = best_model.evaluate(test_data, test_targets)

print("MAE for test data:", test_mae_score)
print("Average absolute error (test MAE * Ystd):", test_mae_score*Ystd)

exit()

