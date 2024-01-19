import numpy as np


# EXAMPLE, TODO: create a proper structure
def enter_long_position(price):
    # this will be API call to the exchange
    return True, price


def close_long_position(price):
    # this will be API call to the exchange
    return False, price


def trailing_stop_loss(current_price, entry_price, stop_loss_percent):
    # this will be set in the API call when made to the exchange
    trailing_stop = entry_price * (1 - stop_loss_percent / 100)
    if current_price < trailing_stop:
        return close_long_position(trailing_stop)
    return True, entry_price


def trade_signals(signals, threshold, stop_loss_percent):
    """
    Long Position (Buy Bitcoin):
    - If the model generates a positive signal (e.g., yb = 3) on a given day, it suggests a potential positive return range for the next day.
    - If the signal remains positive on the following day (i.e., it's still above the threshold), consider taking a long position in Bitcoin by buying.

    Short Position (Sell Bitcoin):
    - If the model generates a negative signal (e.g., yb = -3) on a given day, it suggests a potential negative return range for the next day.
    - If the signal remains negative on the following day (i.e., it's still below the threshold), consider taking a short position in Bitcoin by selling.

    No Trade (Keep Cash):
    - If the model generates a neutral signal (e.g., yb = 0) on a given day, it suggests that there is no strong indication for a specific direction in the next day's returns.
    - In this case, it's advisable to keep cash and not take any positions in Bitcoin.
    """

    long_position = False
    entry_price = 0

    for signal in signals:
        if signal > threshold:
            if not long_position:
                long_position, entry_price = enter_long_position(np.random.uniform(50, 60))
            else:
                pass  # Continue monitoring the position
        elif signal < 0 or (signal == 0 and long_position):
            if long_position:
                long_position, entry_price = close_long_position(np.random.uniform(60, 70))
        if long_position and signal < threshold:
            long_position, entry_price = trailing_stop_loss(np.random.uniform(55, 65), entry_price, stop_loss_percent)

    return long_position, entry_price


def simulate_return(df, budget=1000, leverage=10):
    initial_budget = budget
    final_budget = []

    for current_leverage in range(1, leverage + 1):
        for i in range(len(df) - 1):
            current_position = df.iloc[i]['predicted_position']
            next_day_percentage_change = df.iloc[i + 1]['percentage_change'] / 100

            if current_position == 'Long':
                budget *= (1 + next_day_percentage_change * current_leverage)
            elif current_position == 'Short':
                budget *= (1 - next_day_percentage_change * current_leverage)
            # No action for 'Cash'
        print(f'Leverage: {current_leverage}, Budget: {budget}')
        final_budget.append({"leverage": current_leverage, "budget": budget})
        budget = initial_budget
    return final_budget
