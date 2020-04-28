import os
from models.model_generator import Model
from data.process_data import ProcessData
import json
from data.plot_data import *
import datetime as dt
from sklearn.model_selection import KFold


def main():
    configs = json.load(open('config.json', 'r'))
    epochs = str(configs['training']['epochs'])
    model_config = configs['models'][0]
    dim = model_config['layers'][0]['input_dim']
    results_path = model_config['results']

    if not os.path.exists(model_config['saved_models']): os.makedirs(model_config['saved_models'])
    if not os.path.exists(results_path): os.makedirs(results_path)

    data = ProcessData(configs)

    if configs['data']['normalise']:
        data.normalize_data()

    x_train, y_train, train_dates = data.get_labeled_data(train_data=True, single_step=True)
    x_test, y_test, test_dates = data.get_labeled_data(train_data=False, single_step=True)

    model = Model()
    model.build_model(configs)

    if configs["training"]["load_model"]:
        model.load_model(os.path.join(model_config['saved_models'], configs["training"]["load_model_name"]))
        plot_path = os.path.join(results_path, '%s-%s-e%s-dim%s-pred.png' %
                                 (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(model_config['name']),
                                  epochs, dim))

        evaluate(model, x_test, y_test, test_dates, plot_path, data)
    else:
        k_fold_split = configs['training']['k_fold_split']
        k = 1
        for train_index, validate_index in KFold(k_fold_split).split(x_train):
            folded_x_train, folded_x_validate = x_train[train_index], x_train[validate_index]
            folded_y_train, folded_y_validate = y_train[train_index], y_train[validate_index]
            history = model.train(folded_x_train, folded_y_train, folded_x_validate, folded_y_validate, k, configs)

            plot_loss_path = os.path.join(results_path, '%s-%s-e%s-k%s-dim%s-loss.png' %
                                          (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(model_config['name']),
                                           epochs, str(k), dim))
            plot_train_history(history, "Training and Validation Loss", plot_loss_path)
            plot_pred_path = os.path.join(results_path, '%s-%s-e%s-k%s-dim%s-pred.png' %
                                          (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(model_config['name']),
                                           epochs, str(k), dim))
            evaluate(model, x_test, y_test, test_dates, plot_pred_path, data)

            k += 1


def evaluate(trained_model, x_test, y_test, test_dates, save_path, data):
    trained_model.evaluate_model(x_test, y_test)
    predictions = trained_model.predict_point_by_point(x_test[425:713])
    plot_results(data.denormalize_target(predictions), data.denormalize_target(y_test[425:713]),
                 test_dates[425:713], save_path)


if __name__ == '__main__':
    main()
