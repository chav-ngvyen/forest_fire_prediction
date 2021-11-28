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
features = 'ALL';			# Options are 'STFWI,' 'STM,' 'FWI,' 'M,' or 'ALL'

#-----------------------
# GridSearch parameters
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
# Hyperparameters to be optimized?
# Number of hidden layers
# Size of hidden layers
# Dropout
# Regularization

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
print("Mean of log-transformed area:",Ymean)
print("Standard deviation of log-transformed area:",Ystd)

#------------------
# Define model
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
# ---------------
# Save results
# ---------------


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
    


exit()
    
# #------------------
# # K-fold validation
# #------------------

# num_val_samples = len(train_data) // k
# val_mae_list = []
# val_rmse_list = []
# train_mae_list = []
# train_rmse_list = []
# all_mae_scores = []
# all_rmse_scores = []


# for i in range(k):
#     print('processing fold #',i)
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_data = np.asarray(val_data).astype(np.float32)
#     val_targets = np.asarray(train_targets[i * num_val_samples: (i + 1) * num_val_samples])
#     val_targets = np.asarray(val_targets).astype(np.float32)
    
#     partial_train_data = np.concatenate(
#         [train_data[:i * num_val_samples],
#          train_data[(i + 1) * num_val_samples:]],
#         axis = 0).astype(np.int)
    
#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples],
#          train_targets[(i + 1) * num_val_samples:]],
#         axis=0).astype(np.int)
    

#     print("folk #",i,"shape:",partial_train_data.shape)
    
#     model1 = build_model()
#     history = model1.fit(partial_train_data,
#                          partial_train_targets,
#                          epochs = num_epochs,
#                          batch_size=1,
#                          validation_data=(val_data, val_targets),
#                          verbose=1)


#     print("Evaluate model from fold #", i, "on validation data:")
#     val_mae, val_rmse = model1.evaluate(val_data, val_targets, verbose=1)
#     all_mae_scores.append(val_mae)
#     all_rmse_scores.append(val_rmse)

#     # MAE Loss
#     train_mae_history = history.history['loss']
#     train_mae_list.append(train_mae_history)

#     val_mae_history = history.history['val_loss']
#     val_mae_list.append(val_mae_history) 

#     # RMSE  
#     train_rmse_history = history.history['root_mean_squared_error']
#     train_rmse_list.append(train_rmse_history)

#     val_rmse_history = history.history['val_root_mean_squared_error']
#     val_rmse_list.append(val_rmse_history)



# # Calculate the averages
# avg_train_mae_history = [np.mean([x[i] for x in train_mae_list]) for i in range(num_epochs)]
# avg_train_rmse_history = [np.mean([x[i] for x in train_rmse_list]) for i in range(num_epochs)]

# avg_val_mae_history = [np.mean([x[i] for x in val_mae_list]) for i in range(num_epochs)]
# avg_val_rmse_history = [np.mean([x[i] for x in val_rmse_list]) for i in range(num_epochs)]


# Plot them
epochs = range(1, len(avg_train_mae_history) + 1)

plt.figure()
plt.plot(epochs, avg_train_mae_history, 'bo', label='Training MAE')
plt.plot(epochs, avg_val_mae_history, 'r', label='Validation MAE')
plt.title("Training and validation MAE")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.savefig('../Plots/train_val_MAE_'+features+'_OneHot_'+str(ONE_HOT)+'.png')
plt.show()
plt.clf()

plt.plot(epochs, avg_train_rmse_history, 'bo', label='Training RMSE')
plt.plot(epochs, avg_val_rmse_history, 'r', label='Validation RMSE')
plt.title("Training and validation RMSE")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.savefig('../Plots/train_val_MAE_'+features+'_OneHot_'+str(ONE_HOT)+'.png')
plt.show()
plt.close()


# ---------------------------------------------------------------------------- #

print("MAE across all folds:",all_mae_scores)
print("Average MAE:",np.mean(all_mae_scores))
print("Average MAE (unnormalized):", area_std*np.mean(all_mae_scores))

print("RMSE across all folds:",all_rmse_scores)
print("Average RMSE:",np.mean(all_rmse_scores))
print("Average RMSE (unnormalized):", Ystd*np.mean(all_rmse_scores))

test_mae_score, test_rmse_score = model1.evaluate(test_data, test_targets)

print("MAE of model on test data:",test_mae_score)
print("RMSE of model on test data:",test_rmse_score)
print("MAE of model on test data (unnormalized):",test_mae_score*Ystd)
print("RMSE of model on test data (unnormalized):",test_rmse_score*Ystd)

exit()


# ---------------------------------------------------------------------------- #
# Fit the best model after grid search
# We haven't done grid search yet so this is a place holder

# Get a fresh, compiled model.
best_model = build_model()
# Train it on the entirety of the data.
best_model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)

# Fit best_model on test data
test_data = np.asarray(test_data).astype(np.float32)
test_targets = np.asarray(test_targets).astype(np.float32)

test_mse_score, test_mae_score = best_model.evaluate(test_data, test_targets)

print("MAE for test data:", test_mae_score)
print("MSE for test data:", test_mse_score)


print("Average absolute error (test MAE * Ystd):", test_mae_score*Ystd)

      
# Make predictions
Ypred_train = best_model.predict(train_data)
Ypred_test = best_model.predict(test_data)

# Unnormalize
train_targets=Ystd*train_targets+Ymean
test_targets=Ystd*test_targets+Ymean

Ypred_train = Ystd*Ypred_train+Ymean
Ypred_test = Ystd*Ypred_test+Ymean

# Parity plot
plt.figure()
plt.plot(train_targets, Ypred_train, '*', label = 'Training')
plt.plot(test_targets, Ypred_test, 'r*', label = 'Test')
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.title('Parity plot')
plt.legend()
plt.savefig('../Plots/Parity'+features+'_OneHot_'+str(ONE_HOT)+'.png')
plt.show()
exit()

#------------------
# Train longer (50 epochs) and save validation logs at each fold
#------------------

all_mae_histories = []
for i in range(k):
    print('processing fold #',i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_data = np.asarray(val_data).astype(np.float32)
    val_targets = np.asarray(train_targets[i * num_val_samples: (i + 1) * num_val_samples])
    val_targets = np.asarray(val_targets).astype(np.float32)

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis = 0).astype(np.int)
    
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0).astype(np.int)
    
    print(partial_train_data.shape)
    print(partial_train_targets.shape)
    
    model2 = build_model()
    history = model2.fit(partial_train_data,
                         partial_train_targets,
                         validation_data=(val_data, val_targets),
                         epochs = num_epochs,
                         batch_size=1,
                         verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

#------------------
# Plot validation scores
#------------------

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#------------------
# Plot validation scores
#------------------

def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history)

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

test_mse_score, test_mae_score = model2.evaluate(test_data, test_targets)

print("Mean absolute error of model on test data:",test_mae_score)







