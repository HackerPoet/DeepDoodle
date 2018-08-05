import sys, random
import numpy as np
from matplotlib import pyplot as plt
from dutil import *

NUM_EPOCHS = 50
BATCH_SIZE = 20
VALID_RATIO = 25
lr = 0.0008

def plotScores(scores, test_scores, fname, on_top=True):
	plt.clf()
	ax = plt.gca()
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')
	plt.plot(scores)
	plt.plot(test_scores)
	plt.xlabel('Epoch')
	loc = ('upper right' if on_top else 'lower right')
	plt.legend(['Train', 'Test'], loc=loc)
	plt.draw()
	plt.savefig(fname)

#Load data set
print "Loading Data..."
x_train = np.load('x_data.npy').astype(np.float32) / 255.0
y_train = np.load('y_data.npy').astype(np.float32) / 255.0
num_samples = x_train.shape[0]
print "Loaded " + str(num_samples) + " Samples."

print "Attaching more channels..."
x_train = add_pos(x_train)

#Split data
split_ix = num_samples/VALID_RATIO
x_test = x_train[:split_ix]
y_test = y_train[:split_ix]
x_train = x_train[split_ix:]
y_train = y_train[split_ix:]

print "Shuffling..."
np.random.seed(0)
rng_state = np.random.get_state()
np.random.shuffle(x_train)
np.random.set_state(rng_state)
np.random.shuffle(y_train)
x_train_mini = x_train[:len(x_train)/VALID_RATIO]
y_train_mini = y_train[:len(y_train)/VALID_RATIO]

###################################
#  Create Model
###################################
print "Loading Keras..."
import os, math
os.environ['THEANORC'] = "./gpu.theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.utils import plot_model
from keras import backend as K
K.set_image_data_format('channels_first')

if False:
	print "Loading Model..."
	model = load_model('Model.h5')
	model.optimizer.lr.set_value(lr)
else:
	print "Building Model..."
	model = Sequential()

	model.add(Conv2D(48, (5, 5), padding='same', input_shape=x_train.shape[1:]))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(96, (5, 5), padding='same'))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(192, (5, 5), padding='same'))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(192, (5, 5), padding='same'))
	model.add(Activation("relu"))
	model.add(UpSampling2D(size=(2,2)))
	model.add(Conv2D(192, (5, 5), padding='same'))
	model.add(Activation("relu"))
	model.add(UpSampling2D(size=(2,2)))
	model.add(Conv2D(96, (5, 5), padding='same'))
	model.add(Activation("relu"))
	model.add(UpSampling2D(size=(2,2)))
	model.add(Conv2D(48, (5, 5), padding='same'))
	model.add(Activation("relu"))
	model.add(Conv2D(3, (1, 1), padding='same'))
	model.add(Activation("sigmoid"))

	model.compile(optimizer=Adam(lr=lr), loss='mse')
	plot_model(model, to_file='model.png', show_shapes=True)

###################################
#  Train
###################################
print "Training..."
train_rmse = []
test_rmse = []

for i in xrange(NUM_EPOCHS):
	model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)

	mse = model.evaluate(x_train_mini, y_train_mini, batch_size=128, verbose=0)
	train_rmse.append(math.sqrt(mse))
	print "Train RMSE: " + str(train_rmse[-1])

	mse = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
	test_rmse.append(math.sqrt(mse))
	print "Test RMSE: " + str(test_rmse[-1])

	model.save('Model.h5')
	print "Saved"

	plotScores(train_rmse, test_rmse, 'Scores.png', True)

print "Done"
