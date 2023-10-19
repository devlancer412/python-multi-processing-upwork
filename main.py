from time import time
import pandas as pd
from talib import SMA
from numba import jit
import numpy as np
from itertools import product
from multiprocessing import Lock, Process, Queue, current_process
import queue  # imported for using queue.Empty exception


@jit(nopython=True)
def calculate_signals(open_prices, close_prices, sma9_prices, sma60_prices, high_prices, low_prices, connection, explosion, stop_loss, profit_target, times, entry_hour):
    shares = 0
    buy_price = 0
    sell_price = 0
    signals = np.zeros(len(close_prices))
    bridge = 0
    active_positions = 0

    for i in range(1, len(close_prices)):
        current_time = times[i]
        entry_time = entry_hour * 60  # convert to minutes past midnight
        exit_time = (entry_hour + 1) * 60

        if sma9_prices[i-1] > sma60_prices[i-1] and sma9_prices[i] < sma60_prices[i]:
            bridge = close_prices[i]

        if (entry_time <= current_time <= exit_time and
            close_prices[i] > sma9_prices[i] and
            sma60_prices[i] - sma9_prices[i] >= connection and
            bridge - close_prices[i] >= explosion and
            shares == 0 and
                active_positions < 1):
            shares = 50
            buy_price = open_prices[i+1]
            signals[i+1] = 1
            active_positions += 1
        elif shares > 0 and low_prices[i] <= buy_price - stop_loss:
            shares = 0
            sell_price = buy_price - stop_loss
            signals[i] = -1
            active_positions -= 1
        elif shares > 0 and high_prices[i] >= buy_price + profit_target:
            shares = 0
            sell_price = buy_price + profit_target
            signals[i] = -1
            active_positions -= 1

    return signals


@jit(nopython=True)
def log_trades_jit(date, signal, open_price, close_price, low_price, high_price, profit_target, stop_loss):
    trade_logs_entry = []
    trade_logs_exit = []
    trade_logs_pnl = []
    trade_logs_entry_time = []
    trade_logs_exit_time = []
    trade_logs_duration = []
    shares = 0
    buy_price = 0
    sell_price = 0
    entry_time = None

    for i in range(len(date)):
        if signal[i] == 1:
            shares = 50
            buy_price = open_price[i]
            entry_time = date[i]

        elif signal[i] == -1 and shares > 0:
            if low_price[i] <= buy_price - stop_loss:
                sell_price = buy_price - stop_loss

            elif high_price[i] >= buy_price + profit_target:
                sell_price = buy_price + profit_target

            exit_time = date[i]
            pnl = (sell_price - buy_price) * shares
            duration = (exit_time - entry_time)

            trade_logs_entry.append(buy_price)
            trade_logs_exit.append(sell_price)
            trade_logs_pnl.append(pnl)
            trade_logs_entry_time.append(entry_time)
            trade_logs_exit_time.append(exit_time)
            trade_logs_duration.append(duration)

            shares = 0
            buy_price = 0

    return trade_logs_entry, trade_logs_exit, trade_logs_pnl, trade_logs_entry_time, trade_logs_exit_time, trade_logs_duration


def load_data():
    data = pd.read_csv('GOLD_M1_2020-2023.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['SMA9'] = SMA(data['Close'], timeperiod=9)
    data['SMA60'] = SMA(data['Close'], timeperiod=60)
    data['Datetime'] = data.index.to_pydatetime()
    data['Time'] = data.index.hour * 60 + data.index.minute

    return data


def backtest_strategy(data, explosion_range, stop_loss_range, profit_target_range, entry_hour_range, tasks_to_accomplish: Queue, tasks_that_are_done: Queue):

    best_params = None
    best_performance = float('-inf')
    best_num_trades = 0  # Add this line
    best_winning_percentage = 0  # Add this line
    best_profit_factor = 0  # Add this line
    best_consecutive_losses = 0  # Add this line
    valid_combinations = 0

    initial_balance = 100000

    while True:
        try:
            connection = tasks_to_accomplish.get_nowait()
            print(str(connection) + ' is doing by ' +
                  current_process().name)
        except queue.Empty:
            break
        else:
            for explosion, stop_loss, profit_target, entry_hour in product(explosion_range, stop_loss_range, profit_target_range, entry_hour_range):
                data['Signal'] = calculate_signals(data['Open'].values,
                                                   data['Close'].values,
                                                   data['SMA9'].values,
                                                   data['SMA60'].values,
                                                   data['High'].values,
                                                   data['Low'].values,
                                                   connection,
                                                   explosion,
                                                   stop_loss,
                                                   profit_target,
                                                   data['Time'].values,
                                                   entry_hour)

                buy_price, sell_price, pnl, entry_time, exit_time, duration = log_trades_jit(data.Datetime.values,
                                                                                             data.Signal.values,
                                                                                             data.Open.values,
                                                                                             data.Close.values,
                                                                                             data.Low.values,
                                                                                             data.High.values,
                                                                                             profit_target,
                                                                                             stop_loss)

                trade_records_df = pd.DataFrame({
                    'Entry Price': buy_price,
                    'Exit Price': sell_price,
                    'PNL': pnl,
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Duration': duration})

                num_trades = len(trade_records_df)

                if num_trades >= 10:
                    winning_trades_percentage = (
                        len(trade_records_df[trade_records_df["PNL"] > 0]) / num_trades) * 100
                    gross_profit = trade_records_df.loc[trade_records_df.PNL > 0, 'PNL'].sum(
                    )
                    gross_loss = abs(
                        trade_records_df.loc[trade_records_df.PNL < 0, 'PNL'].sum())
                    profit_factor = gross_profit / \
                        gross_loss if gross_loss != 0 else float('inf')

                    losing_trades_streaks = (
                        trade_records_df["PNL"] < 0).astype(int)
                    consecutive_losing_trades = (losing_trades_streaks * (losing_trades_streaks.groupby(
                        (losing_trades_streaks != losing_trades_streaks.shift()).cumsum()).cumcount() + 1)).max()

                    if winning_trades_percentage >= 50 and profit_factor >= 3 and consecutive_losing_trades <= 5:
                        valid_combinations += 1
                        cumulative_return = (
                            trade_records_df['PNL'].sum() / initial_balance) * 100

                        if cumulative_return > best_performance:
                            best_params = (connection, explosion,
                                           stop_loss, profit_target, entry_hour)
                            best_performance = cumulative_return
                            best_num_trades = num_trades
                            best_winning_percentage = winning_trades_percentage
                            best_profit_factor = profit_factor
                            best_consecutive_losses = consecutive_losing_trades

                        filename = f'trade_records_connection={connection}_explosion={explosion}_stop_loss={stop_loss}_profit_target={profit_target}_entry_hour={entry_hour}.csv'
                        trade_records_df.to_csv(filename, index=False)

    if best_params is not None:
        tasks_that_are_done.put({
            'performance': best_performance,
            'params': best_params,
            'num_trades': best_num_trades,
            'winning_percentage': best_winning_percentage,
            'profit_factor': best_profit_factor,
            'consecutive_losses': best_consecutive_losses,
            'valid_combinations': valid_combinations
        })


def main():
    start_time = time()

    data = load_data()

    connection_range = range(2, 60)
    explosion_range = range(2, 60)
    stop_loss_range = range(3, 6)
    profit_target_range = range(10, 12)
    entry_hour_range = range(0, 4)

    num_combinations = len(connection_range) * len(explosion_range) * len(
        stop_loss_range) * len(profit_target_range) * len(entry_hour_range)
    print(f'Number of combinations tested: {num_combinations}')

    number_of_processes = 8
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes: list[Process] = []

    for connection in connection_range:
        tasks_to_accomplish.put(connection)

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=backtest_strategy, args=(data, explosion_range, stop_loss_range,
                    profit_target_range, entry_hour_range, tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    best_params = None
    best_performance = float('-inf')
    best_num_trades = 0  # Add this line
    best_winning_percentage = 0  # Add this line
    best_profit_factor = 0  # Add this line
    best_consecutive_losses = 0  # Add this line
    valid_combinations = 0

    while not tasks_that_are_done.empty():
        result = tasks_that_are_done.get()
        if result['performance'] > best_performance:
            best_performance = result['performance']
            best_params = result['params']
            best_num_trades = result['num_trades']
            best_winning_percentage = result['winning_percentage']
            best_profit_factor = result['profit_factor']
            best_consecutive_losses = result['consecutive_losses']
            valid_combinations = result['valid_combinations']

    print(f'Best parameters: {best_params}')
    print(f'Best performance: {best_performance}%')
    print(f'Number of trades for the best parameters: {best_num_trades}')
    print(
        f'Winning percentage for the best parameters: {best_winning_percentage}%')
    print(f'Profit factor for the best parameters: {best_profit_factor}')
    print(
        f'Consecutive losses for the best parameters: {best_consecutive_losses}')
    print(f'Number of valid combinations: {valid_combinations}')

    end_time = time()

    elapsed_time = end_time - start_time
    print(elapsed_time)


if __name__ == "__main__":
    main()
