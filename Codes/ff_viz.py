import seaborn as sns; 

import pandas as pd
import numpy as np
import datetime as dt
import time
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers
import numpy as np
from ann_visualizer.visualize import ann_viz
from keras_visualizer import visualizer
# https://github.com/lordmahyar/keras-visualizer

# ---------------------------------------------------------------------------- #
ff_df = pd.read_csv("../Data/forestfires.csv") # Load as Pandas Dataframe

# Convert month str to number
ff_df['month'] = [dt.datetime.strptime(x, '%b').month for x in ff_df['month'] ]
# Convert day str to number. mon = 0, sun = 6
ff_df['day'] = [time.strptime(x, '%a').tm_wday for x in ff_df['day'] ]

ff_df['area_log'] = np.log(ff_df.area+1)

# ---------------------------------------------------------------------------- #
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})

# Area variable without log transform
sns.histplot(ff_df.area, kde=False, bins=50).set(xlabel='Area', ylabel='Frequency', title='Target variable (area) without log transform')
plt.savefig("../Plots/area_no_log.png")
plt.clf()

# Area variable without log transform
sns.histplot(ff_df.area_log, kde=False, bins=50).set(xlabel='Ln(Area + 1)', ylabel='Frequency', title='Target variable (area) with log transform')
plt.savefig("../Plots/area_with_log.png")
plt.clf()

# ---------------------------------------------------------------------------- #
columns = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain']
df = ff_df[columns].copy()
# Rename columns
df = df.rename(columns={"X": "X grid", "Y": "Y grid", "month": "Month", "day": "Day of week",
                   "FFMC":"Fine Fuel Moisture Code","DMC":"Duff Moisture Code","DC":"Drought Code",
                   "ISI":"Initial Spread Index","RH":"Outside relative humidity (in %)",
                   "temp":"Outside temperature (in C)",
                   "wind":"Outside wind speed (in km/h)",
                   "rain":"Outside rain (in mm/m2)"})
df.hist(bins=20, figsize=(20, 10), layout=(3, 4))
plt.suptitle("Distribution of input features")
plt.savefig("../Plots/input_hist.png")
# ---------------------------------------------------------------------------- #
# 1 hidden layer viz
model = Sequential()
model.add(layers.Dense(16, activation='relu',input_shape=(4,)))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(1))
visualizer(model,filename='../Plots/NN_1_layer',format='png')

# ---------------------------------------------------------------------------- #
# 2 hidden layer viz
model = Sequential()
model.add(layers.Dense(32, activation='relu',input_shape=(4,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))

visualizer(model,filename='../Plots/NN_2_layers',format='png', view=True)

