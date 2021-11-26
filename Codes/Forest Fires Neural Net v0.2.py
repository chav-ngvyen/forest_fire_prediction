#------------------
# Load forest fires dataset
#------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime as dt
import time
import matplotlib.pyplot as plt
#---------------
# User parameters
#---------------

k = 3;						# Number of folds for k-fold cross validation
num_epochs = 5;				# Number of epochs for training
features = 'FWI';			# Options are 'STFWI,' 'STM,' 'FWI,' 'M,' or 'ALL'
ONE_HOT = 0;
TIME_TRANSFORM = 1;         # Not yet implemented
# ---------------------------------------------------------------------------- #

ff_df = pd.read_csv("../Data/forestfires.csv") # Load as Pandas Dataframe
ff_df['logarea'] = np.log(ff_df.area + 1) # Log transform for response variable


# Convert month str to number
ff_df['month'] = [dt.datetime.strptime(x, '%b').month for x in ff_df['month'] ]
# Convert day str to number. mon = 0, sun = 6
ff_df['day'] = [time.strptime(x, '%a').tm_wday for x in ff_df['day'] ]

def time_transform(column):
	max_value = column.max()
	sin_values = [np.sin((2*np.pi*x)/max_value) for x in list(column)]
	cos_values = [np.math.cos((2*np.pi*x)/max_value) for x in list(column)]
	return sin_values, cos_values

if TIME_TRANSFORM == 1:
	day_of_week_sin, day_of_week_cos = time_transform(ff_df['day'])
	month_of_year_sin, month_of_year_cos = time_transform(ff_df['month'])

if ONE_HOT == 1 :
	## Select if using One Hot Encoder for month & day
	# Dummies for day of week
	day_of_week_columns = pd.get_dummies(ff_df['day'])
	ff_df = ff_df.merge(day_of_week_columns, left_index=True, right_index=True)
	# Dummies for month
	month_columns = pd.get_dummies(ff_df['month'])
	ff_df = ff_df.merge(month_columns, left_index=True, right_index=True)
	# Drop the str columns
	ff_df = ff_df.drop(columns=['month','day'])

ff = np.array(ff_df) # Convert to numpy array

area_mean = np.mean(ff[:,12])
area_std = np.std(ff[:,12])

print(area_mean)
print(area_std)

log_area_mean = np.mean(ff[:,13])
log_area_std = np.std(ff[:,13])

print(log_area_mean)
print(log_area_std)


# ---------------------------------------------------------------------------- #
# Hyperparameters to be optimized?
# Number of hidden layers
# Size of hidden layers
# Dropout
# Regularization

# Select the features based on user parameters

if ONE_HOT == 0:
	# Not using One Hot Encoder for month & day
	if features == 'STFWI':
		ff_features = ff[:,0:8]
		ff_response = ff[:,13]
	if features == 'FWI':
		ff_features = ff[:,4:8]
		ff_response = ff[:,12]
	if features == 'M':
		ff_features = ff[:,8:12]
		ff_response = ff[:,13]
	if features == 'STM':
		ff_features = ff[:,[0,1,2,3,8,9,10,11,12]]
		ff_response = ff[:,13]
	if features == 'ALL':
		ff_features = ff[:,0:12]
		ff_response = ff[:,13]  
   
if ONE_HOT == 1:
	# Using One Hot Encoder for month & day
	if features == 'STFWI':
		ff_features = ff[:,[0,1,2,3,4,5,6,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
		ff_response = ff[:,11]
	if features == 'FWI':
		ff_features = ff[:,2:6]
		ff_response = ff[:,11]
	if features == 'M':
		ff_features = ff[:,6:10]
		ff_response = ff[:,11]
	if features == 'STM':
		ff_features = ff[:,[0,1,2,3,4,5,6,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
		ff_response = ff[:,11]
	if features == 'ALL':
		ff_features = ff[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
		ff_response = ff[:,11] 

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
from tensorflow.keras import models
from tensorflow.keras import layers

def build_model():
	model = models.Sequential()
	model.add(layers.Dense(32, activation='relu',input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(32, activation='relu'))
	model.add(layers.Dense(32, activation='relu'))
	model.add(layers.Dense(32, activation='relu'))
	model.add(layers.Dense(32, activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop',
	            loss='mae',
	            metrics=[tf.keras.metrics.RootMeanSquaredError()])
	return model


model_ex = build_model()
model_ex.summary()

#------------------
# K-fold validation
#------------------

num_val_samples = len(train_data) // k
val_mae_list = []
val_rmse_list = []
train_mae_list = []
train_rmse_list = []
all_mae_scores = []
all_rmse_scores = []


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
	

	print("folk #",i,"shape:",partial_train_data.shape)
	
	model1 = build_model()
	history = model1.fit(partial_train_data,
						 partial_train_targets,
						 epochs = num_epochs,
						 batch_size=1,
						 validation_data=(val_data, val_targets),
						 verbose=1)


	print("Evaluate model from fold #", i, "on validation data:")
	val_mae, val_rmse = model1.evaluate(val_data, val_targets, verbose=1)
	all_mae_scores.append(val_mae)
	all_rmse_scores.append(val_rmse)

	# MAE Loss
	train_mae_history = history.history['loss']
	train_mae_list.append(train_mae_history)

	val_mae_history = history.history['val_loss']
	val_mae_list.append(val_mae_history) 

	# RMSE  
	train_rmse_history = history.history['root_mean_squared_error']
	train_rmse_list.append(train_rmse_history)

	val_rmse_history = history.history['val_root_mean_squared_error']
	val_rmse_list.append(val_rmse_history)



# Calculate the averages
avg_train_mae_history = [np.mean([x[i] for x in train_mae_list]) for i in range(num_epochs)]
avg_train_rmse_history = [np.mean([x[i] for x in train_rmse_list]) for i in range(num_epochs)]

avg_val_mae_history = [np.mean([x[i] for x in val_mae_list]) for i in range(num_epochs)]
avg_val_rmse_history = [np.mean([x[i] for x in val_rmse_list]) for i in range(num_epochs)]


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







