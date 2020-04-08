import pandas as pd
import numpy as np


def aggregate_features(df, begin_time_stamp, aggr_time, aggregate_steps):
    five_min_speed_aggr = []
    five_min_vol_aggr = []
    five_min_tt_aggr = []
    missing_indexes = []
    date_list = []
    timestamp_list = []

    for i in range(0, aggregate_steps):
        five_min_result = df.loc[
            (df.time_stamp >= begin_time_stamp) & (df.time_stamp <= begin_time_stamp + aggr_time)]
        five_min_count = five_min_result.shape[0]
        if five_min_count > 0:
            timestamp_list.append(begin_time_stamp + aggr_time)
            five_min_vol_aggr.append(five_min_result.volume_per_hour.mean())
            five_min_speed_aggr.append(five_min_result.speed_kmh.mean())
            five_min_tt_aggr.append(five_min_result.travel_time.mean())
            date_list.append(five_min_result.iloc[five_min_count - 1].date_time)
        else:
            missing_indexes.append(i)

        begin_time_stamp = begin_time_stamp + aggr_time

    return date_list, timestamp_list, five_min_vol_aggr, five_min_speed_aggr, five_min_tt_aggr, missing_indexes


class DataPrepare:

    def __init__(self, start_time, data_range, features_path, labels_path, aggregate_time_range):
        self.start_timestamp = start_time
        self.data_size = data_range
        self.feature_path = features_path
        self.label_path = labels_path
        self.aggr_time = aggregate_time_range

        features_df = pd.read_csv(self.feature_path)
        features_df.volume_per_hour = features_df.loc[:, "volume_per_hour"].replace(to_replace=0, method='ffill')
        features_df.speed_kmh = features_df.loc[:, "speed_kmh"].replace(to_replace=0, method='ffill')
        features_df.travel_time = features_df.loc[:, "travel_time"].replace(to_replace=0, method='ffill')
        features_df.loc[features_df.volume_per_hour > 1800, "volume_per_hour"] = 1800

        measures_df = pd.read_fwf(self.label_path, skiprows=[1], nrows=self.data_size)
        measures_df.Vehicles = measures_df.loc[:, "Vehicles"].replace(to_replace=0, method='ffill')
        measures_df.loc[measures_df["Vehicles"] > 150, ["Vehicles"]] = 150

        labels = measures_df["Vehicles"]
        labels = labels * 12
        labels = labels.to_numpy()
        aggregated_features = aggregate_features(features_df, self.start_timestamp, self.aggr_time, self.data_size)
        # remove missing feature indexes from labels. It's needed because there are some missing values at features.
        labels = np.delete(labels, aggregated_features[4])

        combined_features = list(
            zip(aggregated_features[0], aggregated_features[1], aggregated_features[2], aggregated_features[3],
                aggregated_features[4], labels))

        self.aggregated_features_df = pd.DataFrame(combined_features,
                                                   columns=["date", "timestamp", "volume_per_hour", "speed_kmh",
                                                            "travel_time", "vehicle_count"])

    def save_features_df_to_csv(self, path):
        self.aggregated_features_df.to_csv(path, index=False, header=True)


