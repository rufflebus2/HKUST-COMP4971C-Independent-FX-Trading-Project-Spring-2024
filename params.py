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
import time

import params_and_func as paf

# Data download
# Resolutions: 1/2/5/15/30/60/90m | 1h | 1/5d | 1wk | 1/3mo
# 1d --> Unlimited probably
# 1h --> 730 days
# 2/5/15/90m --> 60 days
# 1m --> 7 days
tickers = ['EURUSD=X','AUDUSD=X','JPYUSD=X','GBPUSD=X','CADUSD=X','BTC-USD','SPY']
end = dt.datetime.now()
start = end - dt.timedelta(days = 730)
interval = '1h'
data = paf.download_data(tickers, start, end, interval)

params = { 
    'ticker': 'BTC-USD',
    'data_type': 'all',
    'interval': interval,
    'split_ratio': 0.5,
    'lookback': 50,
    'sd_mult': 1.5,
    'sl_stop': 0.05,
    'aroon_adj': True, # Aroon indicator
    'lineuthresh': 70,
    'linedthresh': 30,
    'aroon_lb': 24*31,
}
ticker = params['ticker']

# lookback #24, 120, 4
lookback_start = 50 
lookback_end = 150
lookback_step = 50
lookback_points = round((lookback_end-lookback_start)/lookback_step + 1,0)

# sdmult #0.5, 2.5, 0.25
sdmult_start = 2.0
sdmult_end = 3.0
sdmult_step = 0.5
sdmult_points = round((sdmult_end-sdmult_start)/sdmult_step + 1,0)

# sl_stop #0.01, 0.1, 0.01
sl_start = 0.1
sl_end = 0.2
sl_step = 0.05
sl_points = round((sl_end-sl_start)/sl_step + 1,0)

# aroon lookback
aroon_start = 24*15*2
aroon_end = 24*15*2*3
aroon_step = 24*15*2
aroon_points = round((aroon_end-aroon_start)/aroon_step+1,0)

lookback_range = np.linspace(lookback_start, lookback_end, num=int(lookback_points), endpoint=True).astype(int) # np.arange(24, 121, 4) # 24,121,4
sd_mult_range = np.linspace(sdmult_start, sdmult_end, num=int(sdmult_points), endpoint=True) # np.arange(0.5, 3.1, 0.5)
aroon_range = np.linspace(aroon_start, aroon_end, num=int(aroon_points), endpoint=True).astype(int)
sl_range = np.linspace(sl_start, sl_end, num=int(sl_points), endpoint=True)

w_range = sl_range
x_range = lookback_range
y_range = sd_mult_range
z_range = aroon_range

w_name = 'sl_stop'
x_name = 'lookback'
y_name = 'sd_mult'
z_name = 'aroon_lb'

w_list = len(w_range)
x_list = len(x_range)
y_list = len(y_range)
z_list = len(z_range)

print(f'length of {w_name}: {w_list}, length of {x_name}: {x_list}, length of {y_name}: {y_list}, length of {z_name}: {z_list}')
df = pd.DataFrame(columns = [w_name, x_name, y_name, z_name, 'CAGR', 'MDD', 'MAR', 'Return'])

start_total = time.time()
for w in range(w_list):
    for x in range(x_list): 
        for y in range(y_list):
            for z in range(z_list):
                start_aroon = time.time()
                print(w,x,y,z)
                params[w_name] = w_range[w]
                params[x_name] = x_range[x]
                params[y_name] = y_range[y]
                params[z_name] = z_range[z]

                req_price, req_sig, spy_price = paf.getCloseAndSigs(data, params)
                pf = paf.runBacktest(req_sig, req_price, params)

                cagr = paf.getCAGR(pf, len(req_price))
                mdd = paf.getDD(pf)
                mar = paf.getMAR(pf, len(req_price))
                ret = pf.cumulative_returns().iloc[-1]

                newrow = pd.DataFrame([{w_name: w_range[w], x_name: x_range[x], y_name: y_range[y], z_name: z_range[z], 'CAGR': cagr, 'MDD': mdd, 'MAR': mar, 'Return': ret}])
                df = pd.concat([df, newrow], ignore_index=True)
        
                if mar >=2 or cagr >= 0.15:
                    print(f'{w_name}: {w_range[w]}, {x_name}: {x_range[x]}, {y_name}: {y_range[y]}, {z_name}: {z_range[z]} | mar:{round(mar,3)} | cagr: {round(cagr,5)} | mdd: {round(mdd,5)}')
                
                end_aroon = time.time()
                print(f'Ending iteration of {z_range[z]} Aroon Lookback: {end_aroon-start_aroon}')
end_total = time.time()
print(f'Total time taken: {end_total-start_total}')
file_name = f'/filepath/{ticker}_results.csv'
df.to_csv(file_name, index=False)


            

