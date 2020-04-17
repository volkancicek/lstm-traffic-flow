import os
from models.model_generator import Model
from data.process_data import ProcessData
import json
from data.plot_data import *
import datetime as dt


def main():
    configs = json.load(open('config.json', 'r'))
    data = ProcessData(configs)

    if configs['data']['normalise']:
        data.normalize_data()

    x_train, y_train = data.get_labeled_data(train_data=True, single_step=True)
    x_test, y_test = data.get_labeled_data(train_data=False, single_step=True)

    print('Single window of history')
    print(x_train[0])
    print('\n Target to predict')
    print(y_train[0])

    model = Model()
    model.build_model(configs)

    if not os.path.exists(configs['models'][0]['saved_models']): os.makedirs(configs['model']['saved_models'])
    if not os.path.exists(configs['models'][0]['results']): os.makedirs(configs['model']['results'])

    history = model.train(x_train, y_train, x_test, y_test, configs)

    plot_loss_path = os.path.join(configs['models'][0]['results'], '%s-%s-e%s-loss.png' %
                                  (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(configs['models'][0]['name']),
                                   str(configs['training']['epochs'])))
    plot_train_history(history, "Training and Validation Loss", plot_loss_path)

    model.evaluate_model(x_test, y_test)
    predictions = model.predict_point_by_point(x_test)
    plot_pred_path = os.path.join(configs['models'][0]['results'], '%s-%s-e%s-pred.png' %
                                  (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(configs['models'][0]['name']),
                                   str(configs['training']['epochs'])))
    plot_results(data.denormalize_target(predictions), data.denormalize_target(y_test), plot_pred_path)


if __name__ == '__main__':
    main()
