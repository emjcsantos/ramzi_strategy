import sqlite3
import pandas as pd
import yfinance as yf


def setup_database():
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS stock_prices')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            date DATE,
            ticker TEXT,
            adj_close REAL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            dividends REAL,
            stock_splits REAL,
            PRIMARY KEY (date, ticker)
        )
    ''')
    
    conn.commit()
    return conn

def download_stock_data(tickers, start_date):
    conn = setup_database()
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date)
            
            # # Print the columns we got from yfinance
            # print(f"\nColumns from yfinance for {ticker}:")
            # print(df.columns)
            
            # Reset index to make date a column
            df = df.reset_index()
            
            df['ticker'] = ticker
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            })
            
            
            # dividends and stock_splits are 0
            required_columns = ['date', 'ticker', 'adj_close', 'open', 'high', 'low', 
                              'close', 'volume', 'dividends', 'stock_splits']

            # Encountered issues with dividend and stock_splits
            # Add any missing columns with NULL values
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            df = df[required_columns]
            df.to_sql('stock_prices', conn, if_exists='append', index=False)
            print(f"Successfully downloaded and stored data for {ticker}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            # Print full traceback
            import traceback
            print(traceback.format_exc())
    
    conn.close()

def query_data(tickers, start_date=None):
    """
    Query data for multiple tickers from the database
    
    Parameters:
    tickers (str or list): Single ticker string or list of ticker symbols
    start_date (str, optional): Start date in 'YYYY-MM-DD' format
    """
    conn = sqlite3.connect('stock_data.db')
    
    # Convert single ticker to list for consistency
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Create the proper tuple format for SQLite IN clause
    placeholders = ','.join('?' * len(tickers))
    query = f"SELECT * FROM stock_prices WHERE ticker IN ({placeholders})"
    params = tickers.copy()
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    
    query += " ORDER BY date"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df

"""
TO USE
"""
if __name__ == "__main__":
    base_df = pd.read_csv('data/output_patterns.csv')
    ticker_list = base_df.Ticker.unique()
    # ticker_list = ['APA', 'VIPS', 'AUR']
    tickers = ticker_list 
    start_date = '2018-01-01'
    
    # Download and store data
    download_stock_data(tickers, start_date)
    
    # ## to query data
    # stock_data = query_data(ticker_list, '2023-01-01') ### ticker_list and start_date
    # print("\nSample of AAPL data:")

    # print(stock_data.head())