import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pickle

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, sequence_lengths):
        super(TimeSeriesDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.sequence_lengths = sequence_lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.sequence_lengths[idx]

    @staticmethod

    def get_raw_data(weather_path="data/weather_forecast.csv",
                    production_path="data/historical_data.csv"):
        # -------------------------------
        # 1️⃣ Load production data
        # -------------------------------
        df = pd.read_csv(production_path, sep=';')
        df = df.dropna(subset=['DateTime CET'])
        df['date'] = pd.to_datetime(df['DateTime CET'], format='%d/%m/%Y %H:%M', errors='coerce')

        # Convert percentages
        df['Availability'] = df['Availability'].str.replace('%','', regex=False).astype(float) / 100
        df['ODD'] = df['ODD'].str.replace('%','', regex=False).astype(float) / 100

        # Filter invalid rows
        df = df[(df['ODD'] >= 0.01) & (df['Availability'] > 0)]

        # Convert production to numeric
        df['Production [MWh]'] = df['Production [MWh]'].str.replace(',', '.', regex=False)
        df['Production [MWh]'] = pd.to_numeric(df['Production [MWh]'], errors='coerce')

        # Compute corrected production
        df['Pcor'] = df['Production [MWh]'] / df['Availability']

        df.set_index('date', inplace=True)

        # -------------------------------
        # 2️⃣ Load weather data
        # -------------------------------
        dm = pd.read_csv(weather_path, sep=';', low_memory=False)

        # Melt wide to long
        df_weather = dm.melt(
            id_vars=[col for col in dm.columns if not col.startswith("Meteo")],
            value_vars=[col for col in dm.columns if col.startswith("Meteo")],
            var_name="raw_name",
            value_name="value"
        )

        # Extract info from raw_name
        df_weather[['type','site','variabile','provider']] = df_weather['raw_name'].str.extract(
            r'Meteo (.*) Site(\d+) (.*) Provider(\d+)'
        )

        df_weather = df_weather.drop(columns=[col for col in ['DateTime CET.x','DateTime CET.y'] if col in df_weather.columns])

        # Convert types
        for col in ['site','provider','type','variabile']:
            df_weather[col] = df_weather[col].astype('category')

        # Keep only relevant variables
        variables_to_keep = ['Temperature','Wind Speed (km/h)',
                            'Wind Gust Speed (km/h)','Wind Direction']
        df_weather = df_weather[df_weather['variabile'].isin(variables_to_keep)]

        df_weather['value'] = pd.to_numeric(df_weather['value'], errors='coerce')
        df_weather['date'] = pd.to_datetime(df_weather['DateTime CET'], format='%d/%m/%Y %H:%M', errors='coerce')
        df_weather.set_index('date', inplace=True)

        # Pivot to wide format
        df_weather_wide = df_weather.pivot_table(
            index=df_weather.index,
            columns='raw_name',
            values='value'
        ).fillna(0)

        # -------------------------------
        # 3️⃣ Split into train/test based on production availability
        # -------------------------------
        train_index = df['Pcor'].dropna().index
        test_index = df['Pcor'].isna().index

        df_train = df.loc[train_index]
        df_test = df.loc[test_index]

        df_weather_train = df_weather_wide.loc[train_index.intersection(df_weather_wide.index)]
        df_weather_test = df_weather_wide.loc[test_index.intersection(df_weather_wide.index)]

        return df_train, df_weather_train, df_test, df_weather_test


def get_splits(
    length=100,
    horizon=50,
    n_train=200,
    n_val=168,
    n_calibration=168,
    seed=None
):
    df, df_weather, _, df_weather_test = TimeSeriesDataset.get_raw_data()

    # =========================
    # 🔹 TRAIN / VAL / CAL DATA
    # =========================
    X_raw = df_weather.values
    Y_raw = df['Pcor'].values.reshape(-1, 1)

    n_samples = len(X_raw) - length - horizon + 1

    X = np.array([X_raw[i:i+length] for i in range(n_samples)])
    Y = np.array([Y_raw[i+length:i+length+horizon] for i in range(n_samples)])
    sequence_lengths = np.array([length] * n_samples)

    # ❗ time-based split (NO shuffle)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    cal_idx = np.arange(n_train + n_val, n_train + n_val + n_calibration)

    # =========================
    # 🔹 TEST DATA (NO TARGET)
    # =========================
    X_test_raw = df_weather_test.values

    n_test_samples = len(X_test_raw) - length + 1
    X_test = np.array([X_test_raw[i:i+length] for i in range(n_test_samples)])
    sequence_lengths_test = np.array([length] * n_test_samples)

    # =========================
    # 🔹 SCALING
    # =========================
    scaler = StandardScaler()

    # Fit ONLY on train
    X_train_scaled = scaler.fit_transform(
        X[train_idx].reshape(-1, X.shape[2])
    ).reshape(len(train_idx), length, X.shape[2])

    def scale(X_subset):
        return scaler.transform(
            X_subset.reshape(-1, X.shape[2])
        ).reshape(len(X_subset), length, X.shape[2])

    X_val_scaled = scale(X[val_idx])
    X_cal_scaled = scale(X[cal_idx])
    X_test_scaled = scale(X_test)

    # =========================
    # 🔹 DATASETS
    # =========================
    train_dataset = TimeSeriesDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(Y[train_idx]),
        torch.LongTensor(sequence_lengths[train_idx])
    )

    val_dataset = TimeSeriesDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(Y[val_idx]),
        torch.LongTensor(sequence_lengths[val_idx])
    )

    cal_dataset = TimeSeriesDataset(
        torch.FloatTensor(X_cal_scaled),
        torch.FloatTensor(Y[cal_idx]),
        torch.LongTensor(sequence_lengths[cal_idx])
    )

    # ❗ Test has NO Y
    test_dataset = TimeSeriesDataset(
        torch.FloatTensor(X_test_scaled),
        torch.zeros(len(X_test_scaled), horizon, 1),  # dummy
        torch.LongTensor(sequence_lengths_test)
    )

    return train_dataset, val_dataset, cal_dataset, test_dataset