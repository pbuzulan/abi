import numpy as np


def annualized_geometric_average(returns):
    n = len(returns)
    rG = (np.prod(1 + returns) ** (365 / n)) - 1
    return rG * 100  # Convert to percentage


def annualized_arithmetic_average(returns):
    n = len(returns)
    rA = (np.mean(returns) * 365)
    return rA * 100  # Convert to percentage


def annualized_volatility(returns):
    n = len(returns)
    volatility = np.std(returns) * np.sqrt(365)
    return volatility * 100  # Convert to percentage


def information_ratio(returns):
    rA = annualized_arithmetic_average(returns)
    rG = annualized_geometric_average(returns)
    return rA / rG

# # Example usage:
# bitcoin_returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.015]  # Replace with your actual daily returns
# rG = annualized_geometric_average(bitcoin_returns)
# rA = annualized_arithmetic_average(bitcoin_returns)
# volatility = annualized_volatility(bitcoin_returns)
# info_ratio = information_ratio(bitcoin_returns)
#
# print("Annualized Geometric Average Return:", rG)
# print("Annualized Arithmetic Average Return:", rA)
# print("Annualized Volatility:", volatility)
# print("Information Ratio:", info_ratio)
