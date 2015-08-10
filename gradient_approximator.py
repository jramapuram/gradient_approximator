__author__ = 'jramapuram'

import numpy as np
import matplotlib.pyplot as plt

from itertools import islice
from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM

input_size = 256
hidden_size = input_size / 2
max_samples = input_size * 1000
batch_size = 128

def format_lstm_data(x):
    # Need to create a 3d vector [samples, timesteps, input_dim]
    if len(x.shape) < 3:
        x = x[:, np.newaxis, :]
        print 'formatted data to fit LSTM: ', x.shape
    return x

def unformat_lstm_data(x):
    # Need to create a 2d vector [samples, input_dim]
    if len(x.shape) > 2:
        x = np.squeeze(x, (1,))
        print 'unformatted data from LSTM: ', x.shape
    return x

def split(mat, test_ratio):
    train_ratio = 1.0 - test_ratio
    train_index = np.floor(len(mat) * train_ratio)
    if mat.ndim == 3:
        return mat[0:train_index, :, :], mat[train_index + 1:len(mat) - 1, :, :]
    elif mat.ndim == 2:
        return mat[0:train_index, :], mat[train_index + 1:len(mat) - 1, :]
    elif mat.ndim == 1:
        return mat[0:train_index], mat[train_index + 1:len(mat) - 1]
    else:
        print 'dimensionality is not 2 or 1!'
        raise NotImplementedError

# http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def generate_data():
    f = [np.sin(i) for i in range(0, max_samples)]
    g = [np.cos(i) for i in range(0, max_samples)]
    return np.array(f), np.array(g)

# main
func, grad = generate_data()
func = format_lstm_data(np.array([f for f in window(func, input_size)]))
grad = format_lstm_data(np.array([g for g in window(grad, input_size)]))
func, func_test = split(func, 0.3)
grad, grad_test = split(grad, 0.3)
assert(func.shape == grad.shape)
assert(func_test.shape == grad_test.shape)

print 'training shape: %s | target shape: %s' % (func.shape, grad.shape)
print 'test shape: %s | tes target shape: %s' % (func_test.shape, grad_test.shape)

model = Sequential()
model.add(LSTM(input_size, hidden_size, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(hidden_size, input_size, return_sequences=True))
model.compile(loss='mse', optimizer='adam')
model.fit(func, grad, batch_size=batch_size, nb_epoch=2)
score = model.evaluate(func_test, grad_test, batch_size=batch_size)
print 'evaluation score: ', score

fig = plt.figure()
fig.suptitle('gradient_predictions[blue] vs. true_gradients[red]')
plt.plot(model.predict(func_test[0:1, :, :], batch_size=1).flatten())
plt.plot(grad_test[0:1, :, :].flatten(), color='red')
plt.show()
