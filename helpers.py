import os
import warnings
import pandas as pd
from tqdm import tqdm

from data_downloader import query_data
from strategy import BaseStrategy

warnings.filterwarnings('ignore')



def proc_data(tickers):
    df = query_data(tickers)
    df_f = df.copy()

    df_f['date'] = df_f.date.apply(lambda x: x.split(' ')[0])
    df_f['date'] = pd.to_datetime(df_f['date'])
    df_f.set_index('date', inplace=True)
    df_f = df_f[['ticker', 'open', 'high', 'low', 'close']]
    
    return df_f

def proc_indata(df):
    base_df_filtered = df[['Date', 'BreakoutDate', 'HighWaterMark', 'Ticker', 'RunUpPercentage', 'ConsolidationEndDate']]
    base_df_filtered = base_df_filtered.sort_values(by=['Date', 'RunUpPercentage']).reset_index(drop=True)
    base_df_filtered[['Date', 'BreakoutDate', 'ConsolidationEndDate']] = base_df_filtered[['Date', 'BreakoutDate', 'ConsolidationEndDate']].map(lambda x: x.split(' ')[0])

    return base_df_filtered


def process_strategy(samp_data, df_f, cfg):
    print(cfg)
    params = {
            'ticker': samp_data['Ticker'],
            'ma_len': cfg['ma_len'], 
            'risk': cfg['risk'], ### 1% risk per trade
            'initial_portfolio_value': cfg['initial_portfolio_value'],
            'entry_date': samp_data['BreakoutDate'],
            'conso_end': samp_data['ConsolidationEndDate'],
            'entry_price': samp_data['HighWaterMark'],
            'min_sl_distance': cfg['min_sl_distance'] ### set minimum initial SL distance
        }
    
    st = BaseStrategy(df_f, params)
    df = st.proc_strategy()
    if not os.path.exists('results'):
        os.makedirs('results/trades', exist_ok=True)
    if not df.empty:
        df.to_csv(f'results/trades/{params["ticker"]}_trades.csv')
        df[params['ticker']] = df.portfolio_value.pct_change()
        returns = pd.DataFrame(df[params['ticker']])
    else:
        returns = pd.DataFrame()
    
    return returns

def calculate_returns_with_limit(rets_df, limit_trades=5, exclude_columns=None):
    """
    Calculate cumulative returns with a specified limit on trades.
    
    Parameters:
    - rets_df (DataFrame): DataFrame containing returns data.
    - limit_trades (int): Maximum number of trades to include in calculations for each row.
    - exclude_columns (list of str): Columns to exclude from calculations.
    
    Returns:
    - DataFrame: Updated rets_df with 'count' and 'total_rets' columns.
    """
    if exclude_columns is None:
        exclude_columns = ['count', 'total_rets']

    rets_df['count'] = rets_df.count(axis=1)
    rets_df['total_rets'] = 0
    exclude = exclude_columns.copy()

    for index, row in tqdm(rets_df.iterrows(), total=len(rets_df)):
        idx = rets_df.loc[index, ~row.index.isin(exclude)].dropna().index
        exclude.extend(idx[limit_trades:].tolist())
        
        filt_data = rets_df.loc[index, ~row.index.isin(exclude)].dropna().iloc[:limit_trades]
        rets_df.at[index, 'total_rets'] = filt_data.sum()

    # Calculate cumulative returns and save to CSV
    cumulative_returns = (1 + rets_df['total_rets']).cumprod()
    cumulative_returns.to_csv('rets.csv')
    print("Cumulative returns saved to 'rets.csv'")
    
    return rets_df