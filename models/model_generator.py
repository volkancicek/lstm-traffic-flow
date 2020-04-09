import os
import numpy as np
import datetime as dt
from numpy import newaxis
from helpers.timer import Timer
import tensorflow as tf


class Model:

    def __init__(self):
        self.model = tf.keras.models.Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = tf.keras.models.load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_time_steps = layer['input_time_steps'] if 'input_time_steps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(tf.keras.layers.Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(tf.keras.layers.LSTM(neurons, input_shape=(input_time_steps, input_dim),
                                                    return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(tf.keras.layers.Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=['acc'])

        print('[Model] Model Compiled')
        self.model.summary()
        timer.stop()

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size, buffer_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        train_data_single = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data_single = train_data_single.cache().shuffle(buffer_size).batch(batch_size).repeat()

        val_data_single = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data_single = val_data_single.batch(1).repeat()

        save_file_path = os.path.join(save_dir,
                                      '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        model_history = self.model.fit(train_data_single, epochs=epochs, steps_per_epoch=50,  # x_train.shape[0],
                                       validation_data=val_data_single, validation_steps=10)  # x_val.shape[0])
        self.model.save(save_file_path)

        print('[Model] Training Completed. Model saved as %s' % save_file_path)
        print('\n [Model] results :', model_history.history)

        timer.stop()
        return model_history

    def evaluate_model(self, x, y):
        print('\n# Evaluate on test data')
        results = self.model.evaluate(x, y)
        print('test loss :', results)

    def predict_point_by_point(self, data):
        timer = Timer()
        timer.start()
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        timer.stop()

        return predicted