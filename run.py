import os
from models.model_generator import Model
from data.process_data import ProcessData
import json
from data.plot_data import *


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['saved_models']): os.makedirs(configs['model']['saved_models'])

    data = ProcessData(
        os.path.join('data', configs['data']['approach_1']['data_file_name']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        configs['data']['target_column']
    )

    model = Model()
    model.build_model(configs)

    if configs['data']['normalise']:
        data.normalise_data()

    x_train, y_train = data.get_labeled_data(configs['data']['history_size'], configs['data']['target_range'],
                                             configs['data']['step'], train_data=True, single_step=True)
    x_val, y_val = data.get_labeled_data(configs['data']['history_size'], configs['data']['target_range'],
                                         configs['data']['step'], train_data=False, single_step=True)

    print('Single window of history')
    print(x_train[0])
    print('\n Target to predict')
    print(y_train[0])

    history = model.train(x_train, y_train, x_val, y_val, epochs=configs['training']['epochs'],
                          batch_size=configs['training']['batch_size'], buffer_size=configs['training']['buffer_size'],
                          save_dir=configs['model']['saved_models'])

    model.evaluate_model(x_val, y_val)
    predictions = model.predict_point_by_point(x_val)
    plot_results(predictions, y_val)
    plot_train_history(history, "Training and Validation Loss")


if __name__ == '__main__':
    main()
