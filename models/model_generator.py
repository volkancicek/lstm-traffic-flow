import datetime as dt
import os
import numpy as np
import tensorflow as tf

from helpers.timer import Timer


class Model:

    def __init__(self):
        self.model = tf.keras.models.Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = tf.keras.models.load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['models'][0]['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else False
            input_time_steps = layer['input_time_steps'] if 'input_time_steps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            first_layer = layer['first_layer'] if 'first_layer' in layer else False

            if layer['type'] == 'lstm':
                self.model.add(tf.keras.layers.LSTM(neurons, input_shape=(input_time_steps, input_dim),
                                                    activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                                                    return_sequences=return_seq))
            if layer['type'] == 'gru':
                self.model.add(tf.keras.layers.GRU(neurons, input_shape=(input_time_steps, input_dim),
                                                   activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                                                   return_sequences=return_seq))
            if layer['type'] == 'rnn':
                self.model.add(tf.keras.layers.RNN(neurons, input_shape=(input_time_steps, input_dim),
                                                   activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                                                   return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(tf.keras.layers.Dropout(dropout_rate))
            if layer['type'] == 'dense':
                self.model.add(tf.keras.layers.Dense(neurons, activation=activation))

        self.model.compile(loss=configs['models'][0]['loss'], optimizer=configs['models'][0]['optimizer'],
                           metrics=['mse', 'mae'])

        print('[Model] Model Compiled')
        self.model.summary()
        timer.stop()

    def train(self, x_train, y_train, x_val, y_val, configs):
        timer = Timer()
        timer.start()

        epochs = configs['training']['epochs']
        batch_size = configs['training']['batch_size']
        buffer_size = configs['training']['buffer_size']
        model_save_dir = configs['models'][0]['saved_models']
        model_name = configs['models'][0]['name']
        results_save_dir = configs['models'][0]['results']

        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.cache().shuffle(buffer_size)
        train_data = train_data.batch(batch_size, drop_remainder=True).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data = val_data.batch(1).repeat()

        model_path = os.path.join(model_save_dir, '%s-%s-e%s.h5' %
                                  (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), model_name, str(epochs)))
        model_history = self.model.fit(train_data, epochs=epochs, steps_per_epoch=200,
                                       # x_train.shape[0],
                                       validation_steps=50,
                                       # x_val.shape[0],
                                       validation_data=val_data)
        self.model.save(model_path)

        print('[Model] Training Completed. Model saved as %s' % model_path)

        result_path = os.path.join(results_save_dir, '%s-%s-e%s_result.txt' %
                                   (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), model_name, str(epochs)))

        self.save_result(model_history.history, result_path)

        print('\n [Model] results :', model_history.history)

        timer.stop()
        return model_history

    def evaluate_model(self, x, y):
        timer = Timer()
        timer.start()
        print('\n[Model] Evaluate on test data')
        results = self.model.evaluate(x, y)
        print('test loss, mse, mae :', results)
        timer.stop()

    def predict_point_by_point(self, data):
        timer = Timer()
        timer.start()
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        timer.stop()

        return predicted

    def save_result(self, history, path):
        file = open(path, "w")
        file.write(str(history))
        file.close()
