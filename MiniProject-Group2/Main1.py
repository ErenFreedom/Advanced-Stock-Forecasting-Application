import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def run_prediction(start_date_str, end_date_str, data_path):
    
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])

    
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df.loc[mask]

    
    dataset = df_filtered['Open'].values
    dataset = dataset.reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = sc.fit_transform(dataset)

    
    X_train = []
    y_train = []
    for i in range(60, len(dataset_scaled)):
        X_train.append(dataset_scaled[i-60:i, 0])
        y_train.append(dataset_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1)
    ])

    
    model.compile(optimizer='adam', loss='mean_squared_error')

    
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    
    predicted_stock_price = model.predict(X_train)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    
    plt.figure(figsize=(14,7))
    plt.plot(df_filtered['Date'], dataset, color='black', label='Real Apple Stock Price')
    plt.plot(df_filtered['Date'][60:], predicted_stock_price, color='red', label='Predicted Apple Stock Price')
    plt.title('Apple Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Apple Stock Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_path = r'/home/kartik/Desktop/Mini_Project_ML/AAPL.csv'  
    run_prediction('2020-01-01', '2020-12-31', data_path)
