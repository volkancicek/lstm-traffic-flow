import os
from models.model_generator import Model
from data.process_data import ProcessData
import json
from data.plot_data import *
import datetime as dt
from sklearn.model_selection import KFold


def main():
    configs = json.load(open('config.json', 'r'))
    data = ProcessData(configs)

    if configs['data']['normalise']:
        data.normalize_data()

    x_train, y_train = data.get_labeled_data(train_data=True, single_step=True)
    x_test, y_test = data.get_labeled_data(train_data=False, single_step=True)

    model = Model()
    model.build_model(configs)

    if not os.path.exists(configs['models'][0]['saved_models']): os.makedirs(configs['model']['saved_models'])
    if not os.path.exists(configs['models'][0]['results']): os.makedirs(configs['model']['results'])

    k_fold_split = configs['training']['k_fold_split']
    k = 1
    for train_index, validate_index in KFold(k_fold_split).split(x_train):
        folded_x_train, folded_x_validate = x_train[train_index], x_train[validate_index]
        folded_y_train, folded_y_validate = y_train[train_index], y_train[validate_index]
        history = model.train(folded_x_train, folded_y_train, folded_x_validate, folded_y_validate, k, configs)
        plot_loss_path = os.path.join(configs['models'][0]['results'], '%s-%s-e%s-k%s-loss.png' %
                                      (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(configs['models'][0]['name']),
                                       str(configs['training']['epochs']), str(k)))
        plot_train_history(history, "Training and Validation Loss", plot_loss_path)
        model.evaluate_model(x_test, y_test)
        predictions = model.predict_point_by_point(x_test)
        plot_pred_path = os.path.join(configs['models'][0]['results'], '%s-%s-e%s-k%s-pred.png' %
                                      (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(configs['models'][0]['name']),
                                       str(configs['training']['epochs']), str(k)))
        plot_results(data.denormalize_target(predictions), data.denormalize_target(y_test), plot_pred_path)
        k += 1


if __name__ == '__main__':
    main()
