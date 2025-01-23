import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# load companies from JSON file
def load_companies_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# create directory
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# price trends for each company
def plot_price_trends(data_folder, companies_by_sector, output_folder):
    create_directory(output_folder)
    
    for sector, companies in companies_by_sector.items():
        sector_folder = os.path.join(output_folder, sector)
        create_directory(sector_folder)
        
        for company in companies:
            ticker = company['ticker']
            company_name = company['name']
            data_file = os.path.join(data_folder, f"{ticker}.csv")
            
            if os.path.exists(data_file):
                data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
                
                # adjusted close price and moving averages
                plt.figure(figsize=(10, 6))
                plt.plot(data['Adj Close'], label="Adj Close", linewidth=2)
                plt.plot(data['MA50'], label="50-Day MA", linestyle="--")
                plt.plot(data['MA100'], label="100-Day MA", linestyle="--")
                plt.plot(data['MA200'], label="200-Day MA", linestyle="--")
                
                plt.title(f"{company_name} ({ticker}) - Price Trend")
                plt.xlabel("Date")
                plt.ylabel("Price (USD)")
                plt.legend()
                plt.grid()
                
                output_file = os.path.join(sector_folder, f"{ticker}_price_trend.png")
                plt.savefig(output_file)
                plt.close()

# sector performance
def plot_sector_performance(data_folder, companies_by_sector, output_folder):
    create_directory(output_folder)
    
    sector_performance = {}
    
    for sector, companies in companies_by_sector.items():
        sector_returns = []
        for company in companies:
            ticker = company['ticker']
            data_file = os.path.join(data_folder, f"{ticker}.csv")
            
            if os.path.exists(data_file):
                data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
                cumulative_return = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0]) - 1
                sector_returns.append(cumulative_return)
        
        if sector_returns:
            sector_performance[sector] = sum(sector_returns) / len(sector_returns)
    
    plt.figure(figsize=(10, 6))
    plt.bar(sector_performance.keys(), sector_performance.values(), color='skyblue')
    plt.title("Sector Performance Comparison")
    plt.xlabel("Sector")
    plt.ylabel("Average Cumulative Return")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    
    output_file = os.path.join(output_folder, "sector_performance_comparison.png")
    plt.savefig(output_file)
    plt.close()

# volatility
def plot_volatility_analysis(data_folder, companies_by_sector, output_folder):
    create_directory(output_folder)
    
    sector_volatility = {}
    
    for sector, companies in companies_by_sector.items():
        sector_folder = os.path.join(output_folder, sector)
        create_directory(sector_folder)
        
        company_volatility = []
        for company in companies:
            ticker = company['ticker']
            company_name = company['name']
            data_file = os.path.join(data_folder, f"{ticker}.csv")
            
            if os.path.exists(data_file):
                data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
                data['Daily_Volatility'] = data['Adj Close'].pct_change().abs()
                
                # company volatility
                plt.figure(figsize=(10, 6))
                plt.plot(data['Daily_Volatility'], label="Daily Volatility", linewidth=2)
                plt.title(f"{company_name} ({ticker}) - Volatility Analysis")
                plt.xlabel("Date")
                plt.ylabel("Volatility")
                plt.legend()
                plt.grid()
                
                output_file = os.path.join(sector_folder, f"{ticker}_volatility.png")
                plt.savefig(output_file)
                plt.close()
                
                company_volatility.append(data['Daily_Volatility'].mean())
        
        if company_volatility:
            sector_volatility[sector] = sum(company_volatility) / len(company_volatility)
    
    plt.figure(figsize=(10, 6))
    plt.bar(sector_volatility.keys(), sector_volatility.values(), color='orange')
    plt.title("Sector Volatility Comparison")
    plt.xlabel("Sector")
    plt.ylabel("Average Daily Volatility")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    
    output_file = os.path.join(output_folder, "sector_volatility_comparison.png")
    plt.savefig(output_file)
    plt.close()

# correlation matrix companies in sector
def generate_sector_correlation(data_folder, companies_by_sector, output_folder):
    create_directory(output_folder)
    
    for sector, companies in companies_by_sector.items():
        sector_folder = os.path.join(output_folder, sector)
        create_directory(sector_folder)
        
        sector_data = pd.DataFrame()
        for company in companies:
            ticker = company['ticker']
            data_file = os.path.join(data_folder, f"{ticker}.csv")
            
            if os.path.exists(data_file):
                data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
                sector_data[ticker] = data['Adj Close'].pct_change()
        
        sector_data = sector_data.dropna()
        
        if not sector_data.empty:
            correlation_matrix = sector_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"{sector} - Correlation Matrix")
            
            output_file = os.path.join(sector_folder, f"{sector}_correlation_matrix.png")
            plt.savefig(output_file)
            plt.close()

# correlation sectors
def generate_sector_to_sector_correlation(data_folder, companies_by_sector, output_folder):
    create_directory(output_folder)
    
    sector_returns = {}
    
    for sector, companies in companies_by_sector.items():
        sector_daily_returns = []
        for company in companies:
            ticker = company['ticker']
            data_file = os.path.join(data_folder, f"{ticker}.csv")
            
            if os.path.exists(data_file):
                data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
                daily_returns = data['Adj Close'].pct_change()
                sector_daily_returns.append(daily_returns)
        
        if sector_daily_returns:
            sector_returns[sector] = pd.concat(sector_daily_returns, axis=1).mean(axis=1)
    
    sector_returns_df = pd.DataFrame(sector_returns).dropna()
    
    if not sector_returns_df.empty:
        correlation_matrix = sector_returns_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Sector-to-Sector Correlation Matrix")
        
        output_file = os.path.join(output_folder, "sector_correlation_matrix.png")
        plt.savefig(output_file)
        plt.close()

# volume trends
def plot_volume_trends(data_folder, companies_by_sector, output_folder):
    create_directory(output_folder)
    
    for sector, companies in companies_by_sector.items():
        sector_folder = os.path.join(output_folder, sector)
        create_directory(sector_folder)
        
        sector_volume = pd.DataFrame()
        
        for company in companies:
            ticker = company['ticker']
            company_name = company['name']
            data_file = os.path.join(data_folder, f"{ticker}.csv")
            
            if os.path.exists(data_file):
                data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
                
                plt.figure(figsize=(10, 6))
                plt.plot(data['Volume'], label="Volume", linewidth=2)
                plt.title(f"{company_name} ({ticker}) - Volume Trend")
                plt.xlabel("Date")
                plt.ylabel("Volume")
                plt.legend()
                plt.grid()
                
                output_file = os.path.join(sector_folder, f"{ticker}_volume_trend.png")
                plt.savefig(output_file)
                plt.close()
                
                sector_volume[ticker] = data['Volume']
        
        if not sector_volume.empty:
            sector_volume['Average Volume'] = sector_volume.mean(axis=1)
            plt.figure(figsize=(10, 6))
            plt.plot(sector_volume['Average Volume'], label="Average Volume", linewidth=2, color="orange")
            plt.title(f"{sector} - Average Volume Trend")
            plt.xlabel("Date")
            plt.ylabel("Average Volume")
            plt.legend()
            plt.grid()
            
            output_file = os.path.join(output_folder, f"{sector}_average_volume_trend.png")
            plt.savefig(output_file)
            plt.close()

# main
def create_plots(json_file, data_folder, price_trend_folder, sector_performance_folder, 
                 volatility_folder, correlation_matrix_folder, volume_trend_folder):
    companies_by_sector = load_companies_from_json(json_file)
    
    plot_price_trends(data_folder, companies_by_sector, price_trend_folder)
    plot_sector_performance(data_folder, companies_by_sector, sector_performance_folder)
    plot_volatility_analysis(data_folder, companies_by_sector, volatility_folder)
    generate_sector_correlation(data_folder, companies_by_sector, correlation_matrix_folder)
    generate_sector_to_sector_correlation(data_folder, companies_by_sector, correlation_matrix_folder)
    plot_volume_trends(data_folder, companies_by_sector, volume_trend_folder)

json_file = "companies.json"
data_folder = "data"
price_trend_folder = "plots/price_trends"
sector_performance_folder = "plots/sector_performance_comparison"
volatility_folder = "plots/volatility"
correlation_matrix_folder = "plots/correlation_matrix"
volume_trend_folder = "plots/volume_trends"

create_plots(json_file, data_folder, price_trend_folder, sector_performance_folder, 
             volatility_folder, correlation_matrix_folder, volume_trend_folder)
