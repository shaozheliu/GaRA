from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

class InformerDataset:
    def __init__(
        self,
        forecast_horizon: Optional[int] = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
        data_file = "/hy-tmp/data/ETT-small/ETTh1.csv"
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """

        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = data_file
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

        # Read data
        self._read_data()

    def _get_borders(self):
        data_name = self.full_file_path_and_name.split('/')[-1].split('.')[0]
        if data_name == 'ETTh1':
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24
        elif data_name == 'ETTm1':
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4
        elif data_name == 'weather':
            n_train = 8 * 30 * 24 * 5
            n_val = 2 * 30 * 24 * 5
            n_test = 2 * 30 * 24 * 5
        elif data_name == 'traffic':
            n_train = 14 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        val = slice(train_end, val_end)
        test = slice(test_start, test_end)

        return train, val, test

    def _read_data(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders()

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "val":
            self.data = df[data_splits[1], :]
        elif self.data_split == "test":
            self.data = df[data_splits[2], :]
        else:
            raise ValueError(
                f"Dataset setting miss match"
            )

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T

            return timeseries, forecast, input_mask

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T

            return timeseries, input_mask

    def __len__(self):
        if self.task_name == "imputation":
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == "forecasting":
            return (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1



class InformerDataset_test:
    def __init__(
        self,
        forecast_horizon: Optional[int] = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
        data_file = "/hy-tmp/data/ETT-small/ETTh1.csv"
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """

        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = data_file
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

        # Read data
        self._read_data()

    def _get_borders(self):
        data_name = self.full_file_path_and_name.split('/')[-1].split('.')[0]
        if data_name == 'ETTh1':
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24
        elif data_name == 'ETTm1':
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4
        elif data_name == 'weather':
            n_train = 8 * 30 * 24 * 5
            n_val = 2 * 30 * 24 * 5
            n_test = 2 * 30 * 24 * 5
        elif data_name == 'traffic':
            n_train = 14 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24
        elif data_name == 'exchange':
            n_train = 1 * 30 * 12 * 10
            n_val = 1 * 30 * 12 * 3
            n_test = 1 * 30 * 12 * 3

        train_end = n_train
        val_start = train_end - self.seq_len
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        val = slice(val_start, val_end)
        test = slice(test_start, test_end)

        return train, val, test

    def _read_data(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders()

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "val":
            self.data = df[data_splits[1], :]
        elif self.data_split == "test":
            self.data = df[data_splits[2], :]
        else:
            raise ValueError(
                f"Dataset setting miss match"
            )

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T

            return timeseries, forecast, input_mask

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T

            return timeseries, input_mask

    def __len__(self):
        if self.task_name == "imputation":
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == "forecasting":
            return (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1



if __name__ == '__main__':
    # Load data
    train_dataset = InformerDataset_test(data_split="train", random_seed=13, forecast_horizon=192, data_file='/hy-tmp/moment/data/weather/weather.csv')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    for timeseries, forecast, input_mask in train_loader:
        # Move the data to the GPU
        # 在get_item函数中定义好
        timeseries = timeseries.float()  # batch, channel, seq_len
        input_mask = input_mask          # batch, seq_len
        forecast = forecast.float()      # batch, channel, horizon