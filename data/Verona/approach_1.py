from data.prepare_data import DataPrepare
import json


def main():
    configs = json.load(open('..\\..\\config.json', 'r'))
    features_csv_app1 = configs['data']['approach_1']['feature_csv']
    labels_rpt_path = configs['data']['labels_path']
    # start date : 20/12/2019 00:00:00
    start_timestamp = 1576796429000
    # end date: 17/03/2020 06:30:30
    data_size = 25422
    # aggregate per 5 min
    aggregate_time = 300000
    dp = DataPrepare(start_timestamp, data_size, features_csv_app1, labels_rpt_path, aggregate_time)
    dp.save_features_df_to_csv('approach_1_data.csv')


if __name__ == '__main__':
    main()