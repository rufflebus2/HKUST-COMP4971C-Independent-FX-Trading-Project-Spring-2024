import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
import vectorbt as vbt
import datetime as dt

# cd /Users/raphaelbas/Desktop/vscode_workspace/COMP4971C/python_COMP4971C
# python3 params_and_func.py

# FIXED 
FEE_TYPE = 0.00002

# BACKTESTING METRICS
METRICS = [
    'start', 
    'end', 
    'period', 
    'start_value', 
    'end_value', 
    'total_return', 
    'benchmark_return',
    'total_fees_paid',
    'max_dd',
    'max_dd_duration',
    'total_trades',
    'total_closed_trades',
    'total_open_trades',
    'win_rate',
    'best_trade',
    'worst_trade',
    'sharpe_ratio',
]

def download_data(tickers, start, end, interval):
  fx = {}
  for ticker in tickers:
      fx[ticker] = yf.download(tickers=ticker, start=start, end=end, interval=interval)
      if fx[ticker].empty:
          fx.pop(ticker)
  return fx

def regressionLine(x, y):
  coef = np.polyfit(x,y,1)
  return coef

def crossover(df1, df2): # goes above
  cur = df1 > df2
  return cur

def split(closing_price, sigs, ratio = 0.5):
  split_index = int(round(len(closing_price) * ratio, 0))

  insample_sigs = {
    'long_entries': sigs['long_entries'][:split_index],
    'long_exits': sigs['long_exits'][:split_index],
    'short_entries': sigs['short_entries'][:split_index],
    'short_exits': sigs['short_exits'][:split_index],
  }
  insample = {
    'closing_prices': closing_price[:split_index],
    'sigs': insample_sigs,
  }
  
  outsample_sigs = {
    'long_entries': sigs['long_entries'][split_index:],
    'long_exits': sigs['long_exits'][split_index:],
    'short_entries': sigs['short_entries'][split_index:],
    'short_exits': sigs['short_exits'][split_index:],
  }
  outsample = {
    'closing_prices': closing_price[split_index:],
    'sigs': outsample_sigs,
  }

  return insample, outsample, split_index

def get_SPYret(spy_data, datatype):
  split_index = int(round(len(spy_data) * 0.5,0))
  if datatype == 'insample':
    spy_close = spy_data[:split_index]['Close']
  elif datatype == 'outsample':
    spy_close = spy_data[split_index:]['Close']
  else:
    spy_close = spy_data['Close']
  
  spy_ret = spy_close.vbt.to_returns()
  spy_ret = spy_ret.values
  spy_ret[0] = 0
  spy_ret = np.cumprod(1+spy_ret)[-1] - 1
  
  return spy_ret

def aroonUp(close, x_val): # >70 is uptrend, long only
  index = close.index[x_val]
  prices = close.loc[index].values
  period = len(x_val)
  highprice_ind = np.argmax(prices)
  bars_since_high = period - highprice_ind - 1
  aroon_ind = (period - bars_since_high) * 100 / period
  return aroon_ind 

def aroonDown(close, x_val): # >70 is downtrend, shorts only
  index = close.index[x_val]
  prices = close.loc[index].values
  period = len(x_val)
  lowprice_ind = np.argmin(prices)
  bars_since_low = period - lowprice_ind - 1
  aroon_ind = (period - bars_since_low) * 100 / period
  return aroon_ind 

def adj_mask(sig, adj_bool):
  df = pd.DataFrame({'a': sig, 'b': adj_bool})
  df['a and b'] = df[['a', 'b']].all(axis=1)
  masked_list = pd.Series(df['a and b'])
  return masked_list

def LinearRegressionChannel(closing_prices, params):
  close = closing_prices.copy() # For vbt data
  index = closing_prices.index

  bfl_y = []
  U_y = []
  L_y = []

  lookback = params['lookback']
  sd_mult = params['sd_mult']
  all_x = np.arange(lookback, len(close)) # x_ind starting from lookback
  close_comp = pd.Series(close[lookback:], index=index[all_x])

  aroon_adj = params['aroon_adj']
  aroonLong = []
  aroonShort = []
  aroonUptrend = []
  aroonDowntrend = []

  for x,y in enumerate(close):
    if x >= lookback:
      lookback_y = close[x-lookback:x].values # 1st: 0 to 49 inclusive
      lookback_x = np.arange(x-lookback,x)
      price_coef = regressionLine(lookback_x, lookback_y)
      
      if aroon_adj:
        aroon_lb = params['aroon_lb']
        if x >= aroon_lb:
          aroonlookback_x = np.arange(x-aroon_lb,x)
          uptrend = aroonUp(close, aroonlookback_x)
          downtrend = aroonDown(close, aroonlookback_x)
          aroonUptrend.append(uptrend)
          aroonDowntrend.append(downtrend)

          u_line = params['lineuthresh']
          d_line = params['linedthresh']

          if uptrend > u_line and downtrend < d_line: # Long only
            aroonLong.append(True)
            aroonShort.append(False)
          elif downtrend > u_line and uptrend < d_line: # Short only
            aroonLong.append(False)
            aroonShort.append(True)
          else: # Both
            aroonLong.append(True)
            aroonShort.append(True)
        else:
          aroonLong.append(True)
          aroonShort.append(True)

      price_sd = np.std(lookback_y)
      mean_y = price_coef[0] * x + price_coef[1]
      upper_y = mean_y + sd_mult * price_sd
      lower_y = mean_y - sd_mult * price_sd

      # Check if channel is volatile or not OR check if 90% of data is within the channel
      U_y.append(upper_y)
      L_y.append(lower_y)
      bfl_y.append(mean_y)
  
  # Strategy
  upper_series = pd.Series(L_y, index=index[all_x])
  lower_series = pd.Series(U_y, index=index[all_x])
  long_entries = crossover(lower_series, close_comp) # crossover(close_comp, pd.Series(L_y, index=index[all_x])) # crossover(pd.Series(L_y, index=index[all_x]), close_comp)
  long_exits = crossover(close_comp, upper_series) # Change here
  short_exits = crossover(lower_series, close_comp) # crossover(pd.Series(L_y, index=index[all_x]), close_comp) # Change here
  short_entries = crossover(close_comp, upper_series) # crossunder(close_comp, pd.Series(U_y, index=index[all_x]))# crossover(close_comp, pd.Series(U_y, index=index[all_x]))
  
  if aroon_adj:
    aroonLong = pd.Series(aroonLong, index=index[all_x])
    aroonShort = pd.Series(aroonShort, index=index[all_x])
    long_entries = adj_mask(long_entries, aroonLong)
    short_entries = adj_mask(short_entries, aroonShort)

  sigs = {
    'long_entries': long_entries,
    'long_exits': long_exits,
    'short_entries': short_entries,
    'short_exits': short_exits
  }

  return sigs

def getCloseAndSigs(data, params):
  fxdata = data[params['ticker']] # 'EURUSD=X','AUDUSD=X','JPY=X','GBPUSD=X','EURGBP=X'
  closing_prices = fxdata['Close']
  sigs = LinearRegressionChannel(closing_prices, params)

  index_req = fxdata.index[:params['lookback']]
  lookback_cover = pd.Series(False, index=index_req)
  sigs['long_entries'] = pd.concat([lookback_cover, sigs['long_entries']])
  sigs['long_exits'] = pd.concat([lookback_cover, sigs['long_exits']])
  sigs['short_entries'] = pd.concat([lookback_cover, sigs['short_entries']])
  sigs['short_exits'] = pd.concat([lookback_cover, sigs['short_exits']])

  insample, outsample, split_index = split(closing_prices, sigs, params['split_ratio'])

  data_type = params['data_type']
  spy_split = int(round(len(data['SPY'])*params['split_ratio'],0))

  if data_type == 'insample':
      req_price = insample['closing_prices']
      req_sig = insample['sigs']
      spy_price = data['SPY'][:spy_split]['Close']
  elif data_type == 'outsample':
      req_price = outsample['closing_prices']
      req_sig = outsample['sigs']
      spy_price = data['SPY'][spy_split:]['Close']
  elif data_type == 'all':
      req_price = closing_prices
      req_sig = sigs
      spy_price = data['SPY']['Close']

  return req_price, req_sig, spy_price

def runSPY(spy_price, params):
  spy_pf = vbt.Portfolio.from_signals(
      close = spy_price,
      init_cash = 1_000_000,
      freq = params['interval'],
  )
  return spy_pf

def runBacktest(sigs, price, params):
  pf = vbt.Portfolio.from_signals(
      close = price,
      entries = sigs['long_entries'],
      exits = sigs['long_exits'], 
      short_entries = None, #sigs['short_entries'], # short_entries, 
      short_exits = None, # sigs['short_exits'], # short_exits, 
      freq = params['interval'],
      fees = FEE_TYPE,
      lock_cash = True,
      sl_stop = params['sl_stop'],
      init_cash = 1_000_000,
  )
  return pf

def getSharpe(pf):
  hours_per_year = 252*24
  hourly_rf = np.power(1.0467,1/hours_per_year)-1
  retdiff = pf.returns() - hourly_rf
  mult = np.sqrt(hours_per_year)
  sharpe = mult * np.mean(retdiff) / np.std(retdiff)
  return sharpe

def getCAGR(pf,portfolio_period):
  trading_period = 252*24
  final_ret = pf.cumulative_returns().iloc[-1]
  cagr = np.power((1+final_ret),trading_period/portfolio_period)-1
  return cagr

def getMAR(pf, portfolio_period):
  cagr = getCAGR(pf, portfolio_period)
  mdd = pf.max_drawdown()
  mar = cagr/np.abs(mdd)
  return mar

def getDD(pf):
  return pf.max_drawdown()

def plotHM(data, new_y_labels, new_x_labels, title):
  ax = sns.heatmap(data, annot=True, fmt=".4f", cmap="hot")
  ax.set_xticks(np.arange(len(new_x_labels)) + 0.5)
  ax.set_xticklabels(new_x_labels)
  ax.set_yticks(np.arange(len(new_y_labels)) + 0.5)
  ax.set_yticklabels(new_y_labels)

  ax.set_xlabel("sd mult")
  ax.set_ylabel("lookback")
  ax.set_title(title)
  plt.show()

def plotAll(mar_data, cagr_data, mdd_data, new_y_labels, new_x_labels):
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))

  # MAR
  ax1 = sns.heatmap(mar_data, annot=True, fmt=".3f", cmap="hot", ax=axes[0])
  ax1.set_xticks(np.arange(len(new_x_labels)) + 0.5)
  ax1.set_xticklabels(new_x_labels)
  ax1.set_yticks(np.arange(len(new_y_labels)) + 0.5)
  ax1.set_yticklabels(new_y_labels)

  ax1.set_xlabel("sd mult")
  ax1.set_ylabel("lookback")
  ax1.set_title('MAR')

  # CAGR
  ax2 = sns.heatmap(cagr_data*100, annot=True, fmt=".2f", cmap="hot", ax=axes[1])
  ax2.set_xticks(np.arange(len(new_x_labels)) + 0.5)
  ax2.set_xticklabels(new_x_labels)
  ax2.set_yticks(np.arange(len(new_y_labels)) + 0.5)
  ax2.set_yticklabels(new_y_labels)

  ax2.set_xlabel("sd mult")
  ax2.set_ylabel("lookback")
  ax2.set_title('CAGR')

  # MDD
  ax3 = sns.heatmap(mdd_data*100, annot=True, fmt=".2f", cmap="hot", ax=axes[2])
  ax3.set_xticks(np.arange(len(new_x_labels)) + 0.5)
  ax3.set_xticklabels(new_x_labels)
  ax3.set_yticks(np.arange(len(new_y_labels)) + 0.5)
  ax3.set_yticklabels(new_y_labels)

  ax3.set_xlabel("sd mult")
  ax3.set_ylabel("lookback")
  ax3.set_title('MDD')

  # plt.tight_layout()
  plt.show()
