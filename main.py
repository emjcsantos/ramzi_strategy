import sqlite3
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Union, List, Dict
import numpy as np
from dataclasses import dataclass
import multiprocessing

@dataclass
class StrategyParams:
    """Strategy parameters configuration"""
    ma_len: int = 21
    risk: float = 0.01
    initial_portfolio_value: int = 10_000
    min_sl_distance: float = 0.05

class DataProcessor:
    """Handles all data processing operations for stock analysis"""
    
    def __init__(self, db_path: str = 'stock_data.db'):
        self.db_path = db_path
    
    def query_data(self, tickers: Union[str, List[str]], start_date: str = None) -> pd.DataFrame:
        """
        Query data for multiple tickers from the database with connection management
        """
        tickers = [tickers] if isinstance(tickers, str) else tickers
        
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' * len(tickers))
            query = f"""
                SELECT * FROM stock_prices 
                WHERE ticker IN ({placeholders})
                {' AND date >= ?' if start_date else ''}
                ORDER BY date
            """
            params = [*tickers, start_date] if start_date else tickers
            return pd.read_sql_query(query, conn, params=params)
    
    @staticmethod
    def process_price_data(df: pd.DataFrame) -> pd.DataFrame:
        """Process raw price data into the required format"""
        return (df
                .assign(date=lambda x: pd.to_datetime(x['date'].str.split().str[0]))
                .set_index('date')
                [['ticker', 'open', 'high', 'low', 'close']])
    
    @staticmethod
    def process_input_data(df: pd.DataFrame) -> pd.DataFrame:
        """Process input pattern data"""
        date_columns = ['Date', 'BreakoutDate', 'ConsolidationEndDate']
        return (df[['Date', 'BreakoutDate', 'HighWaterMark', 'Ticker', 
                   'RunUpPercentage', 'ConsolidationEndDate']]
                .assign(**{col: lambda x, col=col: x[col].str.split().str[0] 
                         for col in date_columns})
                .sort_values(['Date', 'RunUpPercentage'])
                .drop_duplicates()
                .reset_index(drop=True))

class Strategy:
    """Handles strategy execution and portfolio calculations"""
    
    def __init__(self, price_data: pd.DataFrame, strategy_params: StrategyParams):
        self.df = price_data
        self.strategy_params = strategy_params
        
    def process_strategy(self, trade_data: pd.Series) -> pd.DataFrame:
        """Execute strategy and calculate returns"""
        if self.df.empty:
            return pd.DataFrame()
            
        params = {
            'ticker': trade_data['Ticker'],
            'entry_date': trade_data['BreakoutDate'],
            'conso_end': trade_data['ConsolidationEndDate'],
            'entry_price': trade_data['HighWaterMark'],
            'ma_len': self.strategy_params.ma_len,
            'risk': self.strategy_params.risk,
            'initial_portfolio_value': self.strategy_params.initial_portfolio_value,
            'min_sl_distance': self.strategy_params.min_sl_distance
        }
        
        returns = self.calculate_portfolio_returns(params)
        
        if not returns.empty:
            returns = pd.DataFrame({
                params['ticker']: returns['portfolio_value'].pct_change()
            })
        
        return returns
    
    def calculate_portfolio_returns(self, params: Dict) -> pd.DataFrame:
        """Calculate portfolio returns based on strategy rules"""
        # Filter data for the specific ticker
        ticker_data = self.df[self.df['ticker'] == params['ticker']].copy()
        if ticker_data.empty:
            return pd.DataFrame()

        # Convert dates to datetime
        entry_date = pd.to_datetime(params['entry_date'])
        conso_end = pd.to_datetime(params['conso_end'])
        
        # Filter data from consolidation end to entry date
        mask = (ticker_data.index >= conso_end) & (ticker_data.index <= entry_date)
        trade_data = ticker_data[mask]
        
        if trade_data.empty:
            return pd.DataFrame()
        
        # Calculate moving average
        ticker_data['ma'] = ticker_data['close'].rolling(window=params['ma_len']).mean()
        
        # Initialize portfolio tracking
        portfolio = pd.DataFrame(index=ticker_data.index)
        portfolio['portfolio_value'] = params['initial_portfolio_value']
        
        # Calculate position size and stop loss
        entry_price = params['entry_price']
        risk_amount = params['initial_portfolio_value'] * params['risk']
        
        # Find low point during consolidation for stop loss
        consolidation_low = trade_data['low'].min()
        stop_loss = min(entry_price * (1 - params['min_sl_distance']), consolidation_low)
        
        # Calculate position size based on risk
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return pd.DataFrame()
            
        shares = int(risk_amount / risk_per_share)
        if shares <= 0:
            return pd.DataFrame()
        
        # Calculate returns
        position_value = shares * ticker_data['close']
        initial_cash = params['initial_portfolio_value'] - (shares * entry_price)
        portfolio['portfolio_value'] = initial_cash + position_value
        
        return portfolio

def get_num_workers(n_workers: int) -> int:
    """Convert -1 to number of CPU cores, otherwise return the specified number"""
    if n_workers <= 0:
        return max(1, multiprocessing.cpu_count() - 1)
    return n_workers

def parallel_process_returns(base_df_filtered: pd.DataFrame, 
                           price_data: pd.DataFrame,
                           strategy_params: StrategyParams,
                           limit_trades: int = 5,
                           n_workers: int = -1) -> pd.Series:
    """Process returns calculation in parallel"""
    strategy = Strategy(price_data, strategy_params)
    process_with_data = partial(strategy.process_strategy)
    
    actual_workers = get_num_workers(n_workers)
    
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        results = list(tqdm(executor.map(
            process_with_data, 
            [base_df_filtered.iloc[i] for i in range(len(base_df_filtered))]
        )))
    
    returns_df = pd.concat(results, axis=1)
    
    portfolio_metrics = (returns_df
        .assign(count=lambda x: x.count(axis=1))
        .assign(total_rets=lambda x: calculate_limited_returns(x, limit_trades)))
    
    return (1 + portfolio_metrics['total_rets']).cumprod()

def calculate_limited_returns(df: pd.DataFrame, limit_trades: int) -> pd.Series:
    """Calculate returns with trade limit using vectorized operations"""
    exclude_cols = ['count', 'total_rets']
    data_cols = [col for col in df.columns if col not in exclude_cols]
    
    sorted_returns = np.sort(df[data_cols].values)[:, -limit_trades:]
    return pd.Series(np.nansum(sorted_returns, axis=1), index=df.index)

def main():
    # Configure strategy parameters
    strategy_params = StrategyParams(
        ma_len=21,              
        risk=0.01,              
        initial_portfolio_value=10_000,  
        min_sl_distance=0.05    
    )
    
    processor = DataProcessor()
    
    base_df = pd.read_csv('data/output_patterns.csv')
    ticker_list = base_df['Ticker'].unique()
    
    price_data = processor.process_price_data(
        processor.query_data(ticker_list)
    )
    pattern_data = processor.process_input_data(base_df)
    
    returns = parallel_process_returns(
        base_df_filtered=pattern_data,
        price_data=price_data,
        strategy_params=strategy_params,
        limit_trades=5,
        n_workers=-1
    )
    
    returns.to_csv('rets.csv')
    print("Final portfolio value:", returns.iloc[-1])
    print("\nDaily returns summary:")
    print(returns.pct_change().describe())

if __name__ == '__main__':
    main()