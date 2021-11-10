#------------------
# Load Boston housing dataset
#------------------
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#------------------
# Examine data
#------------------
#train_data.shape

#test_data.shape

#train_targets

#------------------
# Normalize data
#------------------
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
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
print("-----------K = 4 cross-fold validation, training over 10 epochs---------")
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 10
all_scores = []

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






#------------------
# Train longer (50 epochs) and save validation logs at each fold
#------------------
print("-----------K = 4 cross-fold validation, training over 50 epochs---------")
num_epochs = 50
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
# Plot validation scores, excluding first 10 data points
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

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

test_mse_score, test_mae_score = model2.evaluate(test_data, test_targets)

print("Mean absolute error of model on test data:",test_mae_score)






#------------------
# Train over 50 epochs with L2 regularization and save validation logs at each fold
#------------------

from keras import regularizers

#------------------
# Define model
#------------------

def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu',
						   kernel_regularizer=regularizers.l2(0.001),
						   input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu',
						   kernel_regularizer=regularizers.l2(0.001)))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
	return model


print("-----------K = 4 cross-fold validation, training over 50 epochs, with L2 regularization---------")
num_epochs = 50
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

	model3 = build_model()
	history = model3.fit(partial_train_data,
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
# Plot validation scores, excluding first 10 data points
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

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

test_mse_score, test_mae_score = model3.evaluate(test_data, test_targets)

print("Mean absolute error of model on test data:",test_mae_score)







#------------------
# Train over 50 epochs with L2 regularization and bigger layers (128 neurons / layer) and save validation logs at each fold
#------------------

#------------------
# Define model
#------------------

def build_model():
	model = models.Sequential()
	model.add(layers.Dense(128, activation='relu',
						   kernel_regularizer=regularizers.l2(0.001),
						   input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(128, activation='relu',
						   kernel_regularizer=regularizers.l2(0.001)))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
	return model


print("-----------K = 4 cross-fold validation, training over 50 epochs, with L2 regularization and bigger layers (128 neurons / layer)---------")
num_epochs = 50
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

	model4 = build_model()
	history = model4.fit(partial_train_data,
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
# Plot validation scores, excluding first 10 data points
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

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

test_mse_score, test_mae_score = model4.evaluate(test_data, test_targets)

print("Mean absolute error of model on test data:",test_mae_score)





#------------------
# Train over 50 epochs with bigger layers (128 neurons / layer) and save validation logs at each fold
#------------------

#------------------
# Define model
#------------------

def build_model():
	model = models.Sequential()
	model.add(layers.Dense(128, activation='relu',input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
	return model


print("-----------K = 4 cross-fold validation, training over 50 epochs, with bigger layers (128 neurons / layer)---------")
num_epochs = 50
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

	model5 = build_model()
	history = model5.fit(partial_train_data,
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
# Plot validation scores, excluding first 10 data points
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

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

test_mse_score, test_mae_score = model5.evaluate(test_data, test_targets)

print("Mean absolute error of model on test data:",test_mae_score)


print("Model3 (K = 4 cross-fold validation, trained over 50 epochs, with L2 regularization) is selected for achieving the lowest mean absolute error on the test data.")

