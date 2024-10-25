import numpy as np
import pandas as pd

from typing import Dict

class Strategy:
    def pnl(self, entry, exit):
        return (exit - entry) / entry
    def proc_strategy(self):
        pass
    def run(self):
        pass
    
class BaseStrategy(Strategy):
    def __init__(self, 
                 data: pd.DataFrame, 
                 params: Dict):
        
        self.ticker = params['ticker']
        self.entry_date = params['entry_date']
        self.entry_price = params['entry_price']
        self.conso_end = params['conso_end']

        data = data[data.ticker == self.ticker]
        target_index = data.index.get_loc(self.entry_date)
        prev_row_date = data.iloc[target_index - 1].name
    
        self.df = data.loc[prev_row_date:]
        
        self.init_sl_price = self.df.loc[prev_row_date, 'close']
        self.initial_portfolio_value = params['initial_portfolio_value']
        
        
        risk = params['risk'] * params['initial_portfolio_value']
        self.pos_size, self.pos_bool = self.proc_size(risk, params['min_sl_distance'])
        
        
        self.proc_ind() ### set indicators
        
    def proc_ind(self):
        self.df['ema'] = self.df['close'].ewm(span=21, adjust=False).mean()
        atr = np.maximum(self.df.high - self.df.low, 
                         np.maximum(abs(self.df.high - self.df.close.shift()),
                                    abs(self.df.low - self.df.close.shift())
                                   ))
        self.df['Upper_KC_3'] = self.df['ema'] + (3 * atr)
        self.df['Upper_KC_4'] = self.df['ema'] + (4 * atr)

    def proc_size(self, risk, min_distance_pct):
        price_distance = abs(self.entry_price - self.init_sl_price)
        distance_pct = price_distance / self.entry_price
        
        # Check if distance meets minimum requirement
        
        dist_bool = distance_pct < min_distance_pct
        if dist_bool:
            message = (
            f"Invalid distance: {distance_pct:.3%} is below minimum {min_distance_pct:.3%}\n"
            f"Entry: ${self.entry_price:.2f}, Stop: ${self.init_sl_price:.2f}\n"
            f"Minimum required distance: ${self.entry_price * min_distance_pct:.3f}\n"
            f"Ticker: {self.ticker}"
            )
            print(message)
        
        price_diff = abs(self.entry_price - self.init_sl_price)
        return round(risk / price_diff, 2), dist_bool

    def _proc_tp(self, curr_idx, curr_data, strat_dict, tp_num):
        if tp_num == 1:
            tp_col = 'Upper_KC_3'
            update_col = 'tp1'
        elif tp_num == 2:
            tp_col = 'Upper_KC_4'
            update_col = 'tp2'
        ### calculate realized pnl
        realized_change = self.pnl(curr_data['entry_price'], curr_data[tp_col])
        realized_pnl = realized_change * strat_dict['exit_size'] * curr_data['entry_price']
        
        
        strat_dict['realized_pnl'] += realized_pnl
        self.df.loc[curr_idx, 'realized_pnl'] = strat_dict['realized_pnl']

        # update current_size
        strat_dict['curr_size'] -= strat_dict['exit_size']
        self.df.loc[curr_idx, 'position_size'] = strat_dict['curr_size']

        ### calculate unrealized pnl
        unrealized_change = self.pnl(curr_data['entry_price'], curr_data['close'])
        unrealized_pnl = unrealized_change * strat_dict['curr_size'] * curr_data['entry_price']

        strat_dict['unrealized_pnl'] = unrealized_pnl
        self.df.loc[curr_idx, 'unrealized_pnl'] = strat_dict['unrealized_pnl']

        ### update future tp and stop_loss columns
        future_mask = self.df.index >= curr_idx
        self.df.loc[future_mask, update_col] = True
        
        strat_dict[update_col] = True
        strat_dict['stop_loss'] = curr_data['entry_price']
        self.df.loc[future_mask, 'stop_loss'] = curr_data['entry_price']
        

        return strat_dict
    
    def proc_strategy(self):
        # Exit if the position boolean filter is active
        if self.pos_bool:
            return pd.DataFrame()
    
        # Initialize columns and values in the DataFrame
        self.df['entry_price'] = self.entry_price
        self.df['stop_loss'] = self.init_sl_price
        self.df['alloc_size'] = self.pos_size
        self.df['tp1'] = False
        self.df['tp2'] = False
        self.df['portfolio_value'] = self.initial_portfolio_value
        self.df['position_size'] = 0
        self.df['realized_pnl'] = 0
        self.df['unrealized_pnl'] = 0
        self.df['entry_date'] = None
    
        # Set initial strategy parameters
        strat_dict = {
            'entry_date': self.entry_date,
            'tp_size': 0.3,
            'tp1': False,
            'tp2': False,
            'curr_size': 0,
            'exit_size': 0,
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'stop_loss': self.init_sl_price,
        }
    
        # Track if position has been entered
        entered = False
        entry_date = pd.to_datetime(self.entry_date)
        conso_end = pd.to_datetime(self.conso_end)
    
        # Loop through DataFrame rows
        for curr_idx, curr_data in self.df.iterrows():
            
            # Check entry condition if not entered
            if not entered and entry_date <= curr_idx <= conso_end:
                if round(self.entry_price, 4) >= round(curr_data['low'], 4):
                    strat_dict['curr_size'] = self.pos_size
                    self.df.at[curr_idx, 'position_size'] = strat_dict['curr_size']
    
                    # Calculate unrealized P&L
                    unrealized_change = self.pnl(curr_data['entry_price'], curr_data['close'])
                    unrealized_pnl = unrealized_change * strat_dict['curr_size'] * curr_data['entry_price']
                    
                    # Update strategy dictionary and DataFrame
                    strat_dict['unrealized_pnl'] = unrealized_pnl
                    strat_dict['exit_size'] = strat_dict['curr_size'] * strat_dict['tp_size']
                    self.df.at[curr_idx, 'unrealized_pnl'] = strat_dict['unrealized_pnl']
                    self.df.at[curr_idx, 'portfolio_value'] += strat_dict['realized_pnl'] + strat_dict['unrealized_pnl']
                    self.df.at[curr_idx, 'entry_date'] = curr_idx
                    
                    entered = True  # Set as entered
                    print(f'Entered: {self.entry_date} - {curr_idx}')
            
            # Execute position management after entry
            if entered:
                # Adjust Stop Loss if TP1 is hit and EMA condition is met
                if strat_dict['tp1'] and curr_data['ema'] > strat_dict['stop_loss']:
                    future_mask = self.df.index >= curr_idx
                    strat_dict['stop_loss'] = curr_data['ema']
                    self.df.loc[future_mask, 'stop_loss'] = curr_data['ema']
    
                # Check Take Profit 1 condition
                if not strat_dict['tp1'] and curr_data['high'] >= curr_data['Upper_KC_3']:
                    strat_dict = self._proc_tp(curr_idx, curr_data, strat_dict, tp_num=1)
    
                # Check Take Profit 2 condition
                if not strat_dict['tp2'] and curr_data['high'] >= curr_data['Upper_KC_4']:
                    strat_dict = self._proc_tp(curr_idx, curr_data, strat_dict, tp_num=2)
    
                # Check Stop Loss condition
                if curr_data['low'] <= strat_dict['stop_loss']:
                    realized_change = self.pnl(curr_data['entry_price'], strat_dict['stop_loss'])
                    realized_pnl = realized_change * strat_dict['curr_size'] * curr_data['entry_price']
                    strat_dict['realized_pnl'] += realized_pnl
    
                    # Update realized P&L in the DataFrame and trim DataFrame up to exit point
                    self.df.loc[curr_idx, 'realized_pnl'] = strat_dict['realized_pnl']
                    self.df.loc[curr_idx, 'portfolio_value'] += strat_dict['realized_pnl']
                    self.df = self.df[:curr_idx]  # Trim DataFrame at exit point
                    return self.df  # Exit strategy after stop loss is hit
    
                # Calculate and update unrealized P&L for current row
                unrealized_change = self.pnl(curr_data['entry_price'], curr_data['close'])
                unrealized_pnl = unrealized_change * strat_dict['curr_size'] * curr_data['entry_price']
                strat_dict['unrealized_pnl'] = unrealized_pnl
                self.df.loc[curr_idx, 'unrealized_pnl'] = strat_dict['unrealized_pnl']
                self.df.loc[curr_idx, 'realized_pnl'] = strat_dict['realized_pnl']
                self.df.loc[curr_idx, 'position_size'] = strat_dict['curr_size']
                self.df.loc[curr_idx, 'portfolio_value'] += strat_dict['realized_pnl'] + strat_dict['unrealized_pnl']
    
        return pd.DataFrame()