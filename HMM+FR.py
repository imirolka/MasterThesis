import os
import json
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# load companies from JSON file
def load_companies_from_json(file_path):
    with open(file_path, 'r') as file:
        companies_by_sector = json.load(file)
    return companies_by_sector

# load saved preprocessed stock data
def load_saved_data_from_csv(folder):
    loaded_data = {}
    for file_name in os.listdir(folder):
        if file_name.endswith('.csv'):
            ticker = file_name.split('.')[0]
            file_path = os.path.join(folder, file_name)
            data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            loaded_data[ticker] = data
    return loaded_data

# normalize data - scale (0 to 1)
def normalize_stock_data(loaded_data):
    normalized_data = {}
    scaler = MinMaxScaler()
    
    for ticker, data in loaded_data.items():
        if not data.empty:
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            if data.empty:
                print(f"Warning: {ticker} data is empty after NaN removal!")
                continue

            normalized_df = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
            normalized_data[ticker] = normalized_df
    
    return normalized_data

# split data - train and test (0.8)
def split_data(data, train_ratio=0.8):
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data

# augment features
def augment_features(dataframe):
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna() 
    
    if 'Open' in dataframe.columns and 'Close' in dataframe.columns:
        fracclop = np.where(dataframe['Open'] != 0, (dataframe['Close'] - dataframe['Open']) / dataframe['Open'], 0)
    else:
        fracclop = np.nan
    
    if 'High' in dataframe.columns and 'Open' in dataframe.columns:
        frachiop = np.where(dataframe['Open'] != 0, (dataframe['High'] - dataframe['Open']) / dataframe['Open'], 0)
    else:
        frachiop = np.nan

    if 'Open' in dataframe.columns and 'Low' in dataframe.columns:
        fracoplo = np.where(dataframe['Open'] != 0, (dataframe['Open'] - dataframe['Low']) / dataframe['Open'], 0)
    else:
        fracoplo = np.nan

    new_dataframe = pd.DataFrame({'CloseOpen': fracclop, 'HighOpen': frachiop, 'OpenLow': fracoplo})
    new_dataframe.set_index(dataframe.index, inplace=True)
    
    return new_dataframe.dropna()

# train HMM Model
def train_hmm(train_data, n_components=3):
    augmented_data = augment_features(train_data)
    features = augmented_data.dropna()
    X = features.values
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) == 0:
        raise ValueError("Training data contains only NaN or infinite values after cleaning.")

    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=3000, tol=1e-6, init_params="")
    
    # model parameters
    model.startprob_ = np.full(n_components, 1/n_components)
    model.transmat_ = np.full((n_components, n_components), 1/n_components)
    model.means_ = np.mean(X, axis=0) + np.random.rand(n_components, X.shape[1]) * 0.01
    model.covars_ = np.tile(np.var(X, axis=0), (n_components, 1)) + 1e-3
    
    model.fit(X)
    return model

# generate possible sequences
def generate_possible_sequences(data, num_samples=50):
    augmented_data = augment_features(data)
    fracclop = augmented_data['CloseOpen']
    frachiop = augmented_data['HighOpen']
    fracoplo = augmented_data['OpenLow']
    
    sample_space_fracclop = np.linspace(fracclop.min(), fracclop.max(), num_samples)
    sample_space_frachiop = np.linspace(frachiop.min(), frachiop.max(), 10)
    sample_space_fracoplo = np.linspace(fracoplo.min(), frachiop.max(), 10)
    
    possible_outcomes = np.array(list(itertools.product(sample_space_fracclop, sample_space_frachiop, sample_space_fracoplo)))
    return possible_outcomes

# walk-forward prediction
def predict_closing_prices_rolling(model, train_data, test_data, possible_outcomes, num_latent_days=50):
    predicted_close_prices = []
    rolling_train_data = train_data.copy()
    
    for i in tqdm(range(len(test_data))):
        if i < num_latent_days:
            continue  # skiping first days cause not enough data
        
        # rolling window of past num_latent_days
        past_data = test_data.iloc[i - num_latent_days:i]
        augmented_data = augment_features(past_data)
        past_features = augmented_data.dropna().values
        
        outcome_scores = [model.score(np.row_stack((past_features, outcome))) for outcome in possible_outcomes]
        most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)]
        predicted_close_prices.append(test_data.iloc[i]['Open'] * (1 + most_probable_outcome[0]))
        
        # update rolling training data
        rolling_train_data = pd.concat([rolling_train_data, test_data.iloc[[i]]])
        model.fit(augment_features(rolling_train_data).dropna().values)  # update with latest data
    
    return predicted_close_prices

# actual vs predicted prices
def plot_predictions(test_data, predicted_close_prices):
    plt.figure(figsize=(30,10), dpi=80)
    plt.rcParams.update({'font.size': 18})
    x_axis = np.array(test_data.index[len(test_data) - len(predicted_close_prices):], dtype='datetime64[ms]')
    plt.plot(x_axis, test_data.iloc[len(test_data) - len(predicted_close_prices):]['Close'], 'b+-', label="Actual close prices")
    plt.plot(x_axis, predicted_close_prices, 'ro-', label="Predicted close prices")
    plt.legend(prop={'size': 20})
    plt.show()
    
    # error
    ae = abs(test_data.iloc[len(test_data) - len(predicted_close_prices):]['Close'] - predicted_close_prices)
    plt.figure(figsize=(30,10), dpi=80)
    plt.plot(x_axis, ae, 'go-', label="Error")
    plt.legend(prop={'size': 20})
    plt.show()

    actual_prices = test_data.iloc[len(test_data) - len(predicted_close_prices):]['Close']
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_close_prices))
    mape = np.mean(np.abs((actual_prices - predicted_close_prices) / actual_prices)) * 100
    
    print(f"Root Mean Square Error (RMSE) = {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE) = {mape}%")

    print(f"Max error observed = {ae.max()}")
    print(f"Min error observed = {ae.min()}")
    print(f"Mean error observed = {ae.mean()}")


# FR levels
def calculate_fibonacci_levels(data, period=30):
    recent_data = data.iloc[-period:]
    high_price = recent_data['High'].max()
    low_price = recent_data['Low'].min()
    
    levels = {
        '0.0%': high_price,
        '23.6%': high_price - (0.236 * (high_price - low_price)),
        '38.2%': high_price - (0.382 * (high_price - low_price)),
        '50.0%': high_price - (0.500 * (high_price - low_price)),
        '61.8%': high_price - (0.618 * (high_price - low_price)),
        '78.6%': high_price - (0.786 * (high_price - low_price)),
        '100.0%': low_price
    }
    return levels

# closing prices using FR
def predict_closing_prices_with_fibonacci(model, train_data, test_data, possible_outcomes, num_latent_days=50):
    predicted_close_prices = []
    rolling_train_data = train_data.copy()
    
    for i in tqdm(range(len(test_data))):
        if i < num_latent_days:
            continue 
        
        past_data = test_data.iloc[i - num_latent_days:i]
        augmented_data = augment_features(past_data)
        past_features = augmented_data.dropna().values
        
        # fr levels
        fib_levels = calculate_fibonacci_levels(past_data)
        
        # score possible outcomes
        outcome_scores = [model.score(np.row_stack((past_features, outcome))) for outcome in possible_outcomes]
        most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)]
        
        # predict next close price
        predicted_price = test_data.iloc[i]['Open'] * (1 + most_probable_outcome[0])
        
        # adjust if close to fr levels
        for level_name, level_price in fib_levels.items():
            if abs(predicted_price - level_price) / level_price < 0.02:  # 2%
                predicted_price = level_price 
                break
        
        predicted_close_prices.append(predicted_price)
        
        rolling_train_data = pd.concat([rolling_train_data, test_data.iloc[[i]]])
        model.fit(augment_features(rolling_train_data).dropna().values)
    
    return predicted_close_prices

# def plot_all_predictions(test_data, predicted_hmm, predicted_fib):
#     plt.figure(figsize=(15, 6))
#     plt.plot(test_data.index[-len(predicted_hmm):], test_data['Close'].iloc[-len(predicted_hmm):], label="Actual Close Prices", color='blue')
#     plt.plot(test_data.index[-len(predicted_hmm):], predicted_hmm, label="Predicted (HMM Only)", linestyle='dashed', color='red')
#     plt.plot(test_data.index[-len(predicted_fib):], predicted_fib, label="Predicted (HMM + Fibonacci)", linestyle='dotted', color='green')
#     plt.legend()
#     plt.xlabel("Date")
#     plt.ylabel("Stock Price")
#     plt.title("Stock Price Prediction: HMM vs HMM + Fibonacci")
#     plt.show()

def plot_all_predictions(test_data, predicted_hmm, predicted_fib, output_folder, ticker):
    plt.figure(figsize=(15, 6))
    plt.plot(test_data.index[-len(predicted_hmm):], test_data['Close'].iloc[-len(predicted_hmm):], label="Actual Close Prices", color='blue')
    plt.plot(test_data.index[-len(predicted_hmm):], predicted_hmm, label="Predicted (HMM Only)", linestyle='dashed', color='red')
    plt.plot(test_data.index[-len(predicted_fib):], predicted_fib, label="Predicted (HMM + Fibonacci)", linestyle='dotted', color='green')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price Prediction for {ticker}: HMM vs HMM + Fibonacci")
    plt.savefig(os.path.join(output_folder, f"{ticker}_prediction_plot.png"))
    plt.close()


# error metrics
def calculate_accuracy(actual_prices, predicted_prices):
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    return rmse, mape

# main
if __name__ == "__main__":
    data_folder = "data"
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    
    data_dict = load_saved_data_from_csv(data_folder)
    normalized_data = normalize_stock_data(data_dict)
    
    for ticker, data in normalized_data.items():
        print(f"Processing {ticker}...")
        ticker_folder = os.path.join(results_folder, ticker)
        os.makedirs(ticker_folder, exist_ok=True)
        train_data, test_data = split_data(data)
        hmm_model = train_hmm(train_data)
        possible_outcomes = generate_possible_sequences(test_data)
        
        # HMM only
        predicted_close_prices_hmm = predict_closing_prices_rolling(hmm_model, train_data, test_data, possible_outcomes)
        rmse_hmm, mape_hmm = calculate_accuracy(test_data['Close'][len(test_data) - len(predicted_close_prices_hmm):], predicted_close_prices_hmm)
        
        # HMM +FR
        predicted_close_prices_fib = predict_closing_prices_with_fibonacci(hmm_model, train_data, test_data, possible_outcomes)
        rmse_fib, mape_fib = calculate_accuracy(test_data['Close'][len(test_data) - len(predicted_close_prices_fib):], predicted_close_prices_fib)
        

        results_file = os.path.join(ticker_folder, f"{ticker}_results.txt")
        with open(results_file, "w") as f:
            f.write(f"Accuracy Comparison for {ticker}:\n")
            f.write(f"HMM Only - RMSE: {rmse_hmm}, MAPE: {mape_hmm}%\n")
            f.write(f"HMM + Fibonacci - RMSE: {rmse_fib}, MAPE: {mape_fib}%\n")


        # accuracy comparison
        print(f"Accuracy Comparison for {ticker}:")
        print(f"HMM Only - RMSE: {rmse_hmm}, MAPE: {mape_hmm}%")
        print(f"HMM + Fibonacci - RMSE: {rmse_fib}, MAPE: {mape_fib}%")
        
        plot_all_predictions(test_data, predicted_close_prices_hmm, predicted_close_prices_fib, ticker_folder, ticker)

        # plots
        # plot_predictions(test_data, predicted_close_prices_hmm)
        # plot_predictions(test_data, predicted_close_prices_fib)
        # plot_all_predictions(test_data, predicted_close_prices_hmm, predicted_close_prices_fib)

