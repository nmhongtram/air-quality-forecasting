import os
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import holidays


class TimeSeriesDataLoader:
    def __init__(self, file_path, input_size, label_size, offset, train_size, val_size,
                 date_column=None, target_name=None, features_type="M", batch_size=64,
                 nan_threshold=0.00):

        self.input_size = input_size
        self.label_size = label_size
        self.offset = offset if offset > label_size else label_size
        self.train_size = train_size
        self.val_size = val_size
        self.target_name = target_name
        self.features_type = features_type
        self.batch_size = batch_size
        self.nan_threshold = nan_threshold

        # Load and preprocess data
        self.df = pd.read_csv(file_path)

        # Đổi tên cột: loại bỏ đơn vị
        self.df.rename(columns={
            "co µg/m³": "co",
            "pm25 µg/m³": "pm25",
            "no2 µg/m³": "no2",
            "pm10 µg/m³": "pm10"
        }, inplace=True)
        self.df.drop(columns=['datetimeTo_utc', 'datetimeTo_local', 'datetimeFrom_utc', 'pm10'], inplace=True)
        self.df['datetimeFrom_local'] = pd.to_datetime(self.df['datetimeFrom_local'])
        self.df = self.df.set_index('datetimeFrom_local')

        # Tạo lại dãy thời gian liên tục
        full_index = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='1h')
        self.df = self.df.reindex(full_index)

        # Thêm time-based features
        vn_holidays = holidays.Vietnam()
        self.df['hour'] = self.df.index.hour
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['is_holiday'] = self.df.index.normalize().isin(vn_holidays)

        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['dayofweek_sin'] = np.sin(2 * np.pi * self.df['dayofweek'] / 7)
        self.df['dayofweek_cos'] = np.cos(2 * np.pi * self.df['dayofweek'] / 7)
        self.df['is_holiday'] = self.df['is_holiday'].astype(int)

        self.df.drop(columns=['hour', 'dayofweek'], inplace=True)


        # Extract valid sequences
        self.valid_sequences = self._extract_valid_sequences()

        if features_type == 'S':
            self.in_variable = 1
            self.out_variable = 1
        elif features_type == 'M':
            self.in_variable = len(self.df.columns)
            # self.out_variable = len(self.df.columns)
            self.out_variable = 3
        elif features_type == 'MS':
            self.in_variable = len(self.df.columns)
            self.out_variable = 1
        else:
            raise ValueError("Invalid features_type. Choose from 'S', 'M', 'MS'.")

        # Các cột cần transform
        self.cols_to_transform = ["co", "pm25", "no2"]

         # Apply log transform ONLY to the specified columns
        # Note: No scaling is applied in this version
        transformed_df = self.df.copy()

        # Apply log1p transform only to the specified columns
        transformed_df[self.cols_to_transform] = np.log1p(transformed_df[self.cols_to_transform])

        # Replace the original columns with log-transformed values in the main DataFrame
        self.df[self.cols_to_transform] = transformed_df[self.cols_to_transform]

        # Lưu file data đã được xử lý
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
        self.df.to_csv(os.path.join(data_dir, 'processed/processed_data.csv'))

        self.X_train, self.y_train = self._create_dataset(0, int(train_size * len(self.valid_sequences)))
        print(f'X_train.shape = {self.X_train.shape}')
        print(f'y_train.shape = {self.y_train.shape}')
        self.X_val, self.y_val = self._create_dataset(int(train_size * len(self.valid_sequences)),
                                                      int((train_size + val_size) * len(self.valid_sequences)))
        print(f'X_val.shape = {self.X_val.shape}')
        print(f'y_val.shape = {self.y_val.shape}')
        self.X_test, self.y_test = self._create_dataset(int((train_size + val_size) * len(self.valid_sequences)), None)
        print(f'X_test.shape = {self.X_test.shape}')
        print(f'y_test.shape = {self.y_test.shape}')

        self.train_loader = self._create_dataloader(self.X_train, self.y_train)
        self.val_loader = self._create_dataloader(self.X_val, self.y_val)
        self.test_loader = self._create_dataloader(self.X_test, self.y_test)

    def _extract_valid_sequences(self):
        total_len = len(self.df)
        seq_len = self.input_size + self.offset + self.label_size
        valid_chunks = []

        for i in range(0, total_len - seq_len):
            window = self.df.iloc[i:i + seq_len]
            nan_ratio = window.isna().sum().sum() / (seq_len * len(self.df.columns))
            if nan_ratio <= self.nan_threshold:
                valid_chunks.append(i)

        return valid_chunks

    def _create_dataset(self, start_idx, end_idx):
        if end_idx is None:
            end_idx = len(self.valid_sequences)

        features, labels = [], []

        for i in self.valid_sequences[start_idx:end_idx]:
            feat_start = i
            feat_end = feat_start + self.input_size
            label_start = feat_end + self.offset
            label_end = label_start + self.label_size

            if self.features_type == 'S':
                feature = self.df[[self.target_name]].iloc[feat_start:feat_end]
                label = self.df[[self.target_name]].iloc[label_start:label_end]
            elif self.features_type == 'M':
                feature = self.df.iloc[feat_start:feat_end]
                label = self.df.iloc[label_start:label_end][['pm25', 'co', 'no2']]
            elif self.features_type == 'MS':
                feature = self.df.iloc[feat_start:feat_end]
                label = self.df[[self.target_name]].iloc[label_start:label_end]
            else:
                raise ValueError("Invalid features_type.")

            features.append(feature.to_numpy())
            labels.append(label.to_numpy())

        return np.array(features), np.array(labels)

    def _create_dataloader(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
