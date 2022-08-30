# Created by Nicole Zhao

import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import datetime
import matplotlib.pyplot as plt

import sklearn
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.stats import random_correlation


import tensorflow as tf
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials, space_eval
from hyperopt import rand
import sys
import math

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import itertools

def custom_loss(y_true,y_pred):
        returns = tf.reduce_sum(y_true * y_pred)
        loss = -tf.math.log(1+returns)
        return loss

def _train(model, space, X_train, y_train, X_valid=None, y_valid=None, opt='return', validation=True, verbose=False):
    model.compile(loss=space['loss_fn'], optimizer=tf.keras.optimizers.Adam(learning_rate=space['lr'])) #, momentum=space['momentum'], nesterov=space['nesterov']
    if verbose:
        print ('Params testing: ', space)
        model.summary()

    if not validation:
        history = model.fit(X_train, y_train, epochs=150, batch_size = space['batch_size'], verbose=0)
        return history
    else:
        history = model.fit(X_train, y_train, epochs=150, batch_size = space['batch_size'], validation_data=(X_valid, y_valid), verbose=0)

        preds = model.predict(X_valid)
        model_rets = np.array(tf.reduce_sum(preds * y_valid, axis=1))

        if opt == 'return':
            performance = -(tf.reduce_prod(1 + model_rets, axis=0) - 1)
        elif opt == 'sharpe':
            performance = -sharpe(model_rets)
        else:
            performance = - sortino(model_rets)
        return performance, history
    

class Trainer:
    def __init__(self, space, data, opt, normalize, return_data): #
        self.space = space
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = data.get_train_test_data(normalize=normalize)
        self.data = data
        # self.X_train_valid, self.y_train_valid = data.get_train_valid()
        if normalize:
            self.data_type = 'norm'
        if return_data:
            self.data_type = 'ret'
        else:
            self.data_type = 'raw'
            
        self.current_best = math.inf
        self.opt = opt
        self.normalize = normalize
        self.return_data = return_data
        self.nn_type = ''

    def get_model(self, space, load=False):
        pass

    def save_weights(self, model):
        model.save_weights(f'Trained Models/{self.nn_type}_{self.data_type}_{self.opt}')

    def train(self, space, load=False):        
        model = self.get_model(space, load)

        performance, history = _train(model, space, self.X_train, self.y_train, self.X_valid, self.y_valid, validation=True) 

        if performance < self.current_best:
            self.current_best = performance
            model.save_weights(f'Trained Models/{self.nn_type}')
            pd.DataFrame.from_dict({'Param':space.keys(), 'Value':space.values()}).to_csv(f'Trained Models/{self.nn_type}.csv', index=False, sep=';')
            with open(f'{self.nn_type}_best.txt', 'w') as f:
                f.write(str(float(self.current_best)))

        sys.stdout.flush()
        return {
            'loss': performance,
            'status': STATUS_OK,
            'model': model,
            'history': history,
            }

    def tune(self, max_evals):
        trials = Trials()
        best = fmin(
            fn=self.train,
            space=self.space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
            )
        hyperparams = space_eval(self.space, best)
        model = self.get_model(hyperparams)
        # history_valid = _train(model, hyperparams, self.X_train_valid, self.y_train_valid, validation=False)
        history = trials.best_trial['result']['history']
        time = datetime.datetime.now().strftime('%d_%H_%M')
        model.save_weights(f'Trained Models/{self.nn_type}_{self.data_type}_{self.opt}_{time}')
        return {
            'Hyperparams': hyperparams,
            'Opt': self.opt,
            'Normalize': self.normalize,
            'Return_data': self.return_data,
            'time': time,
            'History': history
        }
        
class CNNTrainer(Trainer):
    def __init__(self, space, data, opt, normalize=True, return_data=False):
        super().__init__(space, data, opt, normalize, return_data)
        
        self.nn_type = 'CNN'

    def get_model(self, space, load=False):
        if load:
            df_model = pd.read_csv('Trained Models/CNN.csv',sep=';')
            space = pd.Series(df_model.Value.values, index=df_model.Param).to_dict()
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2], 1)),
            tf.keras.layers.Conv2D(space['Nodes1'], (int(space['filter1_size1']), int(space['filter1_size2'])), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(space['Reg1'])),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),
            tf.keras.layers.Conv2D(space['Nodes2'], (int(space['filter2_size1']), int(space['filter2_size2'])), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(space['Reg2'])),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(space['Nodes4'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(space['Reg4'])),
            tf.keras.layers.Dropout(rate=space['dropout']),
            tf.keras.layers.Dense(self.data.size, activation="softmax")
        ])
        if load:
            model.load_weights('Trained Models/CNN')
            with open('CNN_best.txt') as f:
                self.current_best = f.read()
        return model

class LSTMTrainer(Trainer):
    
    def __init__(self, space, data, opt, normalize, return_data):
        super().__init__(space, data, opt, normalize, return_data)
        self.nn_type = 'LSTM'

    def get_model(self, space,load = False):
        if load:
            df_model = pd.read_csv('Trained Models/LSTM.csv', sep=';')
            space = pd.Series(df_model.Value.values, index=df_model.Param).to_dict()
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.data.n_past, self.data.size)),
            tf.keras.layers.LSTM(space['Nodes1'], recurrent_dropout=space['Dropout1'], return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(space['Reg1'])),
            tf.keras.layers.LSTM(space['Nodes2'], recurrent_dropout=space['Dropout2'], kernel_regularizer=tf.keras.regularizers.l2(space['Reg2'])),
            tf.keras.layers.Dense(self.data.size, activation="softmax")
        ])
        if load:
            model.load_weights('Trained Models/LSTM')
            with open('LSTM_best.txt') as f:
                self.current_best = f.read()
        return model

class RNNTrainer(Trainer):
    
    def __init__(self, space, data, opt, normalize, return_data):
        super().__init__(space, data, opt, normalize, return_data)
        self.nn_type = 'RNN'

    def get_model(self,space,load = False):
        if load:
            df_model = pd.read_csv('Trained Models/RNN.csv', sep=';')
            space = pd.Series(df_model.Value.values, index=df_model.Param).to_dict()
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.data.n_past, self.data.size)),
            tf.keras.layers.SimpleRNN(space['Nodes1'], recurrent_dropout=space['Dropout1'], return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(space['Reg1'])),
            tf.keras.layers.SimpleRNN(space['Nodes2'], recurrent_dropout=space['Dropout2'], kernel_regularizer=tf.keras.regularizers.l2(space['Reg2'])),
            tf.keras.layers.Dense(self.data.size, activation="softmax")
        ])

        if load:
            model.load_weights('Trained Models/RNN')
            with open('RNN_best.txt') as f:
                self.current_best = f.read()
        return model
