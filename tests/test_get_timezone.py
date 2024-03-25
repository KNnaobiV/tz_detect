import datetime
import os
import sys
import math
import pytz

import pandas as pd
import numpy as np
import pytest

sys.path.append(".")

from backend.backend_script import get_timezone as g_tz

@pytest.fixture
def test_folder():
    folder = os.path.join(os.path.dirname(__file__), "data",)
    return folder


@pytest.fixture
def tz_csv(test_folder):
    """file with tz aware datetime objects"""
    filename = os.path.join(
        test_folder, "CADCHF_2016_ASK_UTC.csv"
        )
    return filename


@pytest.fixture
def no_tz_csv(test_folder):
    filename = os.path.join(test_folder, "CADCHF_BID_sydney.csv")
    return filename


@pytest.fixture#(scope="module")
def tz_df(tz_csv):
    df = g_tz.open_csv(tz_csv)
    return df


@pytest.fixture#(scope="module")
def no_tz_df(no_tz_csv):
    df = g_tz.open_csv(no_tz_csv)
    return df

@pytest.fixture
def diff_instrument_df(test_folder):
    df = g_tz.open_csv(os.path.join(test_folder, "EURGBP_tehran.csv"))
    return df

@pytest.fixture
def df_slice():
    begin_date = "2023/03/05"
    df = pd.DataFrame({
        "open": [
            0.72481, 0.72464, 0.72456, 0.72451, 0.72446, 0.72451, 0.72428, 
            0.72423, 0.72385, 0.72399,
            ],
        "high": [
            0.72481, 0.72470, 0.72456, 0.72451, 0.72446, 0.72451, 0.72428, 
            0.72423, 0.72405, 0.72408,
            ],
        "low": [
            0.72457, 0.72445, 0.72456, 0.72451, 0.72444, 0.72430, 0.72425,
            0.72380, 0.72385, 0.72399,
            ],
        "close": [
            0.72458, 0.72445, 0.72456, 0.72451, 0.72444, 0.72430, 0.72426,
            0.72383, 0.72405, 0.72408,
            ],
        "datetime": pd.date_range(
            begin_date, periods=10, tz="Asia/Tokyo", freq="1T"
        )
    })
    
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df

@pytest.fixture
def z_score_df(df_slice):
    df = df_slice
    open_diff = [
        1.515413, 0.910774, 0.626263, 0.448391, 0.270519, 0.448391, 
        -0.369737, -0.547609, -1.899142, -1.401142
    ]
    high_diff = [
        1.629071, 1.170646, 0.587558, 0.379206, 0.170854, 0.379206,
        -0.579115, -0.787467, -1.537435, -1.412275
    ]
    low_diff = [
        1.073835, 0.641423, 1.037962, 0.857736, 0.605335, 0.100961,
        -0.079265, -1.701081, -1.520855, -1.016266
    ]
    close_diff = [
        1.52476, 0.605696, 1.068491, 0.858152, 0.563578, -0.025070,
        -0.193532, -2.002105, -1.076764, -0.950411
    ]
    
    df["open_diff"] = open_diff
    df["high_diff"] = high_diff
    df["low_diff"] = low_diff
    df["close_diff"] = close_diff
    return df

def test_get_filename_pass(no_tz_csv):
    expected = "CADCHF_BID_sydney.csv"
    actual = g_tz.get_filename(no_tz_csv)
    assert expected == actual

def test_get_filename_no_file():
    """write assert raises"""


def test_is_csv_pass(tz_csv):
    assert g_tz.is_csv(tz_csv) == True


def test_is_csv_fail():
    assert g_tz.is_csv("textfile.txt") == False


def test_open_csv(tz_csv, no_tz_csv):
    tz_file = g_tz.open_csv(tz_csv)
    no_tz_file = g_tz.open_csv(no_tz_csv)
    columns = [
        "datetime", "open", "high", "low", "close", "delta", "year", "week"
        ]
    assert list(tz_file) == columns
    assert list(no_tz_file) == columns


def test_is_column_tz_aware(tz_df, no_tz_df):
    assert g_tz.is_column_tz_aware(tz_df) == True
    assert g_tz.is_column_tz_aware(no_tz_df) == False


def test_is_column_tz_aware_raises_attributeerror(tz_df):
    with pytest.raises(AttributeError):
        g_tz.is_column_tz_aware(tz_df, "high")


def test_get_common_unique_for_col():
    pass


def test_calc_diff(df_slice, z_score_df):
    actual = g_tz.calc_diff(df_slice)
    expected = z_score_df
    assert np.array_equal(actual, expected)


def test_get_unusual_diff_behaviour(z_score_df):
    actual = g_tz.get_unusual_diff_behaviour(z_score_df, 3)
    print(actual)
    expected = [
        (8, -1.351533, 'open_diff'), (9, 0.498, 'open_diff'), 
        (6, -0.818128, 'open_diff'), (5, 0.177872, 'open_diff'), 
        (6, -0.958321, 'high_diff'), (5, 0.208352, 'high_diff'), 
        (8, -0.749968, 'high_diff'), (9, 0.12516, 'high_diff'), 
        (7, -1.621816, 'low_diff'), (9, 0.504589, 'low_diff'), 
        (5, -0.504374, 'low_diff'), (2, 0.396539, 'low_diff'), 
        (7, -1.808573, 'close_diff'), (8, 0.925341, 'close_diff'), 
        (1, -0.919064, 'close_diff'), (2, 0.462795, 'close_diff')
    ]
    assert actual == expected

def test_calc_delta(df_slice):
    tz = pytz.timezone("Asia/Tokyo")
    #print(df_slice)
    df_copy = df_slice[:]
    df_copy["datetime"] = pd.date_range("2023-03-05 11:00:00", periods=10, freq="1T")
    tz_time = df_slice["datetime"][0]
    no_tz_time = df_copy["datetime"][0]
    #no_tz_time = no_tz_df["datetime"][0]
    print(tz_time, no_tz_time)
    actual = g_tz.calc_delta(tz_time, no_tz_time)
    assert actual == datetime.timedelta(hours=11)


def test_get_idx(tz_df, no_tz_df):
    expected = datetime.timedelta(hours=11)
    actual = g_tz.get_idx(tz_df, no_tz_df)

    assert actual == expected