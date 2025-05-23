import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from openaq import OpenAQ
from functools import reduce
import holidays

class OpenAQProcessor:
    def __init__(self, API_KEY):
        self.api = OpenAQ(API_KEY)

    def get_data_for_prediction(self, location_id=2161290):
        # Vì offset là 24h nên khung thời gian có thể dự đoán sẽ từ thời điểm hiện tại đến 48 giờ sau.
        sensors = self.api.locations.sensors(locations_id = 2161290)
        sensors_dict = sensors.dict()

        sensor_name_map = {
            sensor['id']: sensor['name']
            for sensor in sensors_dict['results']
        }

        dataframes = []


        for sensor in sensors_dict['results']:
            sensor_id = sensor['id']
            sensor_name = sensor_name_map[sensor_id]

            # Thời gian hiện tại (UTC)
            datetime_last = datetime.now(timezone.utc)

            # Thời gian hiện tại trừ đi 150 giờ
            datetime_first = datetime_last - timedelta(hours=150)

            # Định dạng lại chuỗi thời gian nếu cần thiết để phù hợp với API OpenAQ
            # Ví dụ: '2023-10-27T10:00:00Z'
            datetime_first_str = datetime_first.strftime("%Y-%m-%dT%H:%M:%SZ")
            datetime_last_str = datetime_last.strftime("%Y-%m-%dT%H:%M:%SZ")

            print(f"datetime_first: {datetime_first_str}")
            print(f"datetime_last: {datetime_last_str}")

            all_measurements = []
            page = 1
            limit = 1000        # Cần 96h để dự đoán 48h kế tiếp

            try:
                response = self.api.measurements.list(
                    sensors_id=sensor_id,
                    data='hours',
                    datetime_from=datetime_first,
                    datetime_to=datetime_last,
                    page=page,
                    limit=limit
                )
                response = response.dict()
            except Exception as e:
                print(f"Error for sensor {sensor_id} page {page}: {e}")
                break

            results = response.get('results', [])

            # Tạo DataFrame cho sensor từ các trường phù hợp với cấu trúc response mới
            df = pd.DataFrame([{
                'datetimeFrom_local': r['period']['datetime_from']['local'],
                sensor_name: r['value']
            } for r in results])

            dataframes.append(df)

        # Gộp các DataFrame theo datetime
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['datetimeFrom_local'], how='outer'), dataframes)

        # Sắp xếp theo thời gian
        df_merged = df_merged.sort_values(by='datetimeFrom_local')

        return df_merged


    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Đổi tên cột: loại bỏ đơn vị
        df.rename(columns={
            "co µg/m³": "co",
            "pm25 µg/m³": "pm25",
            "no2 µg/m³": "no2",
            "pm10 µg/m³": "pm10"
        }, inplace=True)
        df.drop(columns=['pm10'], inplace=True)
        df['datetimeFrom_local'] = pd.to_datetime(df['datetimeFrom_local'])
        df = df.set_index('datetimeFrom_local')

        # Tạo lại dãy thời gian liên tục
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
        df = df.reindex(full_index)

        # Xử lý missing values dùng forward fill
        df.fillna(method='ffill', inplace=True)

        # Các cột cần transform
        cols_to_transform = ["co", "pm25", "no2"]

        # Apply log transform ONLY to the specified columns
        transformed_df = df.copy()
        transformed_df[cols_to_transform] = np.log1p(transformed_df[cols_to_transform])
        df[cols_to_transform] = transformed_df[cols_to_transform]

        # Thêm time-based features
        vn_holidays = holidays.Vietnam()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['is_holiday'] = df.index.normalize().isin(vn_holidays)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['is_holiday'] = df['is_holiday'].astype(int)

        df.drop(columns=['hour', 'dayofweek'], inplace=True)

        return df

