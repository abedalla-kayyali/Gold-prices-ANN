#! /usr/bin/python3

import os
import numpy as np
from sklearn import preprocessing

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = BASE_PATH + '/Data/gold_prices.csv'

MIN_YEAR = 1950
MAX_YEAR = 2018
MONTHS_PER_YEAR = 12

def synthetic_data():
    """Testing data. Produces a straight line from 0 to 100 for input.
    For target line is offset by 10.
    
    Returns:
        [type] -- [description]
    """

    # Inputs
    x = np.arange(100)
    # Output sine wave
    y = x + 10
    y = np.expand_dims(y, axis=1)
    # Inputs reshaping
    x = np.expand_dims(x, axis=1)
    # Bias column
    bias = np.ones([100, 1])
    # Append bias
    x = np.append(bias, x, axis=1)
    # Normalize data
    x = preprocessing.minmax_scale(x)
    y = preprocessing.minmax_scale(y)
    return x, y

def _norm_dates(date):
    """[INTERNAL] Normalize the string dates to floats
    
    Arguments:
        date {string} -- Date value ("year-month")
    
    Returns:
        float -- Normalized date value
    """

    # Separate date and year
    year, month = date.split('-')
    # Total months of data
    tot_months = (MAX_YEAR - MIN_YEAR) * MONTHS_PER_YEAR
    # Years passed since beginning of data
    years_passed = (int(year) - MIN_YEAR)
    # Months passed since beginning of data
    months_passed = int(month) + (years_passed * 12)
    # Normalize b/w 0 and 1
    months_norm = months_passed / tot_months
    return months_norm

def get_gold_prices():
    """Read and parse the gold price dataset
    Available at https://datahub.io/core/gold-prices#data
    
    Returns:
        Tuple -- Dates and prices normalized (dates, prices)
    """

    # Empty holders for data
    dates = list()
    prices = list()
    # Read the data file
    with open(DATA_PATH) as data_file:
        # For each row
        for line in data_file:
            # Seperate data
            if not line.strip() == '':
                date, price = line.split(',')
                dates.append(_norm_dates(date.strip()))
                prices.append(float(price.strip()))
    # Convert to numpy arrays
    inputs = np.array(dates).astype(np.float32)
    outputs = np.array(prices).astype(np.float32)
    # Normalize prices
    outputs = preprocessing.minmax_scale(outputs)
    return inputs, outputs
