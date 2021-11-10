#------------------
# Load forest fires dataset
#------------------
import pandas as pd
import numpy as np

ff_df = pd.read_csv("forestfires.csv") # Load as Pandas Dataframe
ff_df['logarea'] = np.log(ff_df.area + 1) # Log transform for response variable
ff = np.array(ff_df) # Convert to numpy array

#---------------
# User parameters
#---------------

k = 10;						# Number of folds for k-fold cross validation
num_epochs = 30;			# Number of epochs for training
features = 'FWI';			# Options are 'STFWI,' 'STM,' 'FWI,' and 'M'

# Hyperparameters to be optimized?
# Number of hidden layers
# Size of hidden layers
# Dropout
# Regularization

# Select features to be used as inputs for neural network
if features == 'STFWI':
	ff_features = ff[:,0:8]
	ff_response = ff[:,13]
if features == 'FWI':
	ff_features = ff[:,4:8]
	ff_response = ff[:,13]
if features == 'M':
	ff_features = ff[:,8:12]
	ff_response = ff[:,13]
if features == 'STM':
	ff_features = ff[:,[0,1,2,3,8,9,10,11,12]]
	ff_response = ff[:,13]


#---------------
# Split into training and test data
#---------------

np.random.seed(12345)
indices = np.random.permutation(ff.shape[0])
CUT = round(0.8*ff.shape[0])
train_idx, test_idx = indices[:CUT], indices[CUT:]
train_data, test_data = ff_features[train_idx,:], ff_features[test_idx,:]
train_targets, test_targets = ff_response[train_idx], ff_response[test_idx]

#------------------
# Examine data
#------------------

train_data.shape
test_data.shape

train_targets.shape
test_targets.shape

#------------------
# Normalize data
#------------------
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.astype('float32').std(axis=0)
train_data /= std

test_data -= mean
test_data /= std



#------------------
# Define model
#------------------
from keras import models
from keras import layers

def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
	return model


#------------------
# K-fold validation
#------------------

num_val_samples = len(train_data) // k
all_scores = []

for i in range(k):
	print('processing fold #',i)
	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

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

	model1 = build_model()
	model1.fit(partial_train_data,
		  partial_train_targets,
		  epochs = num_epochs,
		  batch_size=1,
		  verbose=0)

	val_mse, val_mae = model1.evaluate(val_data, val_targets, verbose=0)
	all_scores.append(val_mae)

print("Mean absolute error for each fold:", all_scores)
np.mean(all_scores)


test_mse_score, test_mae_score = model1.evaluate(test_data, test_targets)

print("Mean absolute error of model on test data:",test_mae_score)


print(partial_train_data)

exit()




#------------------
# Train longer (50 epochs) and save validation logs at each fold
#------------------

all_mae_histories = []
for i in range(k):
	print('processing fold #',i)
	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

	partial_train_data = np.concatenate(
		[train_data[:i * num_val_samples],
		 train_data[(i + 1) * num_val_samples:]],
		axis = 0)
	partial_train_targets = np.concatenate(
		[train_targets[:i * num_val_samples],
		 train_targets[(i + 1) * num_val_samples:]],
		axis=0)

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







