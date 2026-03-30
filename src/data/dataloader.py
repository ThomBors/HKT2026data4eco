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
        # Load production
        df = pd.read_csv(production_path,sep=';')
        df = df.dropna()
        df['date'] = pd.to_datetime(df['DateTime CET'], format='%d/%m/%Y %H:%M', errors='coerce')
        df['Availability'] = df['Availability'].str.replace('%','', regex=False).astype(float) / 100
        df['ODD'] = df['ODD'].str.replace('%','', regex=False).astype(float) / 100
        df = df[(df['ODD'] >= 0.01) & (df['Availability'] > 0)]
        df['Production [MWh]'] = pd.to_numeric(df['Production [MWh]'], errors='coerce')
        df['Pcor'] = df['Production [MWh]'] / df['Availability']

        # Load weather
        dm = pd.read_csv(weather_path, sep=';', low_memory=False)
        df_weather = dm.melt(
            id_vars=[col for col in dm.columns if not col.startswith("Meteo")],
            value_vars=[col for col in dm.columns if col.startswith("Meteo")],
            var_name="raw_name",
            value_name="value"
        )

        df_weather[['type','site','variabile','provider']] = df_weather['raw_name'].str.extract(
            r'Meteo (.*) Site(\d+) (.*) Provider(\d+)'
        )

        df_weather = df_weather.drop(columns=[col for col in ['DateTime CET.x','DateTime CET.y'] if col in df_weather.columns])

        for col in ['site','provider','type','variabile']:
            df_weather[col] = df_weather[col].astype('category')

        variables_to_keep = ['Temperature','Irradiation','Wind Speed (km/h)',
                                'Wind Gust Speed (km/h)','Wind Direction','DateTime CET']



        df_weather = df_weather[df_weather['variabile'].isin(variables_to_keep)]
        df_weather['value'] = pd.to_numeric(df_weather['value'], errors='coerce')
        df_weather_wide = df_weather.pivot_table(
            index='DateTime CET',  # or your time index
            columns='raw_name',
            values='value'
        )
        df_weather_wide = df_weather_wide.fillna(0) 


        return df, df_weather_wide
    
def get_splits(length=100, horizon=50, conformal=True, n_train=200, n_calibration=100, n_test=80, seed=None):
    df, df_weather = TimeSeriesDataset.get_raw_data()
    
    # Merge production and weather here if needed
    # For simplicity, assume we only use Pcor as the target
    # and weather variables as features
    # Convert to pivot table: rows = time, columns = variables
    X_raw = df_weather.values
    Y_raw = df['Pcor'].values.reshape(-1,1)

    # Make sequences
    n_samples = len(X_raw) - length - horizon + 1
    X = np.array([X_raw[i:i+length] for i in range(n_samples)])
    Y = np.array([Y_raw[i+length:i+length+horizon] for i in range(n_samples)])
    sequence_lengths = np.array([length]*n_samples)

    # Shuffle indices
    perm = np.random.RandomState(seed).permutation(n_samples)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train+n_calibration]
    test_idx = perm[n_train+n_calibration:n_train+n_calibration+n_test]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx].reshape(-1, X.shape[2])).reshape(len(train_idx), length, X.shape[2])
    X_cal_scaled = scaler.transform(X[cal_idx].reshape(-1, X.shape[2])).reshape(len(cal_idx), length, X.shape[2])
    X_test_scaled = scaler.transform(X[test_idx].reshape(-1, X.shape[2])).reshape(len(test_idx), length, X.shape[2])

    train_dataset = TimeSeriesDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(Y[train_idx]), torch.LongTensor(sequence_lengths[train_idx]))
    cal_dataset = TimeSeriesDataset(torch.FloatTensor(X_cal_scaled), torch.FloatTensor(Y[cal_idx]), torch.LongTensor(sequence_lengths[cal_idx]))
    test_dataset = TimeSeriesDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(Y[test_idx]), torch.LongTensor(sequence_lengths[test_idx]))

    return train_dataset, cal_dataset, test_dataset