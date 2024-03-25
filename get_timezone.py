import ntpath
import os
import argparse
import math
import statistics
import pytz
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import polars as pl

AWARE_DF_TZ = None

def get_filename(file_path):
    """Takes a file and returns the basename"""
    try:
        filename = ntpath.basename(file_path)
    except FileNotFoundError:
        raise
    return filename


def is_csv(filename):
    if get_filename(filename).endswith(".csv"):
        return True
    else: return False


def open_csv(filename):
    """
    Opens csv file as a dataframe
    
    Returns
    -------
    df: pd.DataFrame
    """
    cols = ["datetime", "open", "high", "low", "close",]
    df_chunks = pd.read_csv(filename, chunksize=100000, engine="c")
    df = pd.concat(df_chunks)
    #polars_df = pl.read_csv(filename)
    #df_as_array = polars_df.to_numpy()
    #df = pd.DataFrame(df_as_array)
    if len(list(df)) == 5:
        df.set_axis(cols, axis=1, inplace=True)
        try:
            df["datetime"] = pd.to_datetime(
                df["datetime"], format="%d.%m.%Y %H:%M:%S.%f"
                )
        except ValueError:
            df["datetime"] = pd.to_datetime(
                df["datetime"], format="%Y-%m-%d %H:%M:%S.%f"
                )
    if len(list(df)) == 7:
        dtypes = {
            "datetime": np.datetime64, "open": np.float32, "high": np.float32, 
            "low": np.float32, "close": np.float32,
        }
        df["datetime"] = pd.to_datetime(
            df[list(df)[0]] + df[list(df)[1]], format="%m/%d/%Y%H:%M:%S"
            )
        cols_to_move = [list(df)[0], list(df)[1], list(df)[6], "datetime"]
        df = df[
            ["datetime"] 
            + [col for col in df.columns if col not in cols_to_move]
            ]
        df.set_axis(cols, axis=1, inplace=True)

    df["year"] = df["datetime"].dt.year
    df["week"]  = df["datetime"].dt.isocalendar().week
    return df


def is_column_tz_aware(df, column_name="datetime"):
    """
    Checks if a dataframe's column is timezone aware.

    Returns
    -------
    bool
    """
    try:
        if df[column_name].dt.tz is not None:
            AWARE_DF_TZ = df[column_name].dt.tz
            return True
        else: return False
    except AttributeError:
        raise


def get_common_unique_for_col(df1, df2, col):
    """
    Returns unique values common to both data frames for the specified 
    col.

    Parameters
    ----------
    df1: pd.DataFrame
    df2: pd.DataFrame
    col: str
        Column name

    
    Returns
    -------
    unique_for_col: list
        list of unique values for the specified column
    """
    unique_for_col = []
    unique_for_col_in_df1 = df1[col].unique()
    unique_for_col_in_df2 = df2[col].unique()

    for value in unique_for_col_in_df1:
        if value in unique_for_col_in_df2:
            unique_for_col.append(value)
    return unique_for_col


def get_dfs_slices_with_common_weeks(df1, df2):
    """
    Returns dfs with year and week values common to both df arguments
    """
    unique_weeks = []
    unique_years = get_common_unique_for_col(df1, df2, "year")
    unique_weeks = get_common_unique_for_col(df1, df2, "week")

    df1_unique_weeks_for_year = df1.loc[
        (df1["year"].isin(unique_years)) 
        & (df1["week"].isin(unique_weeks[1:]))
    ]
    df2_unique_weeks_for_year = df2.loc[
        (df2["year"].isin(unique_years)) 
        & (df2["week"].isin(unique_weeks[1:]))
    ]

    return df1_unique_weeks_for_year, df2_unique_weeks_for_year


def calc_diff(df):
    for col in df.columns[1:5]:
        col_diff = col + '_diff'
        df[col_diff] = df[col] - df[col].shift(1)
        #col_mean = df[col].mean()
        #col_std = df[col].std(ddof=0)
        #df[col_zscore] = (df[col] - col_mean) / col_std
        #df[col_zscore] = pd.to_numeric(df[col_zscore], downcast="float")
    return df


def get_unusual_diff_behaviour(df, chk_range=30):
    """Gets indexes where there are unusal zscore behaviour"""
    indexes = []
    for col in df.columns:
        if col.endswith('diff'):
            diffs = df[col].diff().tolist()
            sorted_diffs = sorted(diffs)
            for i in range(chk_range):
                diff = sorted_diffs[i]
                diff2 = sorted_diffs[-i]
                if not math.isnan(diff):
                    indexes.append((diffs.index(diff), round(diff, 6), col))
                if not math.isnan(diff2):
                    indexes.append((diffs.index(diff2), round(diff2, 6), col))
    return indexes


def get_idx(df1, df2):
    deltas = []
    comp_df1, comp_df2 = get_dfs_slices_with_common_weeks(df1, df2)
    comp_df1_with_diff = calc_diff(comp_df1)
    comp_df2_with_diff = calc_diff(comp_df2)
    df1_comp_idxs = get_unusual_diff_behaviour(comp_df1_with_diff)
    df2_comp_idxs = get_unusual_diff_behaviour(comp_df2_with_diff)
    for idx, value in zip(df1_comp_idxs, df2_comp_idxs):
        if idx[-1] == value[-1]:
            dt1 = comp_df1.iat[idx[0], 0]
            dt2 = comp_df2.iat[value[0], 0]
            if calc_delta(dt1, dt2) < timedelta(hours=24):
                deltas.append(calc_delta(dt1, dt2))
    predicted_tz = statistics.mode(deltas)

    days = int(predicted_tz.components.days)
    hours = int(predicted_tz.components.hours)
    minutes = int(predicted_tz.components.minutes)
    if predicted_tz.components.days:
        hours = 24 - hours

    if hours > 0: timezone_info = f"+{hours}:{minutes}"
    elif hours < 0: timezone_info = f"-{hours}:{minutes}"

    return timezone_info


def calc_delta(t1, t2):
    """Returns the timedelta between two datetime objects"""
    tz_info = None
    try:
        tz = t2.utcoffset()
        tz_aware = t2.tz_convert(None) + t2.dst()
        tz_unaware = t1
    except TypeError:
        tz = t1.utcoffset()
        tz_aware = t1.tz_convert(None) + t1.dst()
        tz_unaware = t2

    if tz_aware > tz_unaware:
        tz_info = -( tz_aware - tz_unaware - tz)
    else:
        tz_info = tz_unaware - tz_aware - tz
    return tz_info


def set_tz(df, tz_info):
    """Can be taken out if not needed"""
    if tz_info in pytz.all_timezones:
        df["datetime"] = pd.to_datetime(
            df["datetime"].dt.tz_localize(tz_info)
        )
        return df
    else:
        raise ValueError(f"{tz_info} is not a valid timezone")
    """elif len(tz_info.split(":")) == 2:
                    location_info = tz_info.split(":")[0][0]
                    hour = int(tz_info.split(":")[0][1:])
                    minutes = int(tz_info.split(":")[-1])
            
                    if hour >= 11:
                        raise ValueError(
                            f"{tz_info} does not belong to a valid timezone. "
                            "Hour value cannot be greater than 11"
                        )
                    if minutes != 30:
                        raise ValueError(
                            f"{tz_info} does not belong to a valid timezone. "
                            f"Expected a minute value of {30}."
                        )
                    df["datetime"] = df["datetime"] + timezone.timedelta(
                        hours=hours, minutes=minutes)
            
                else: return f"{tz_info} is not a valid timezone"""



parser = argparse.ArgumentParser(
    description="Detects timezone information of a timeseries"
    )
parser.add_argument("-g", "--get_timezone", nargs=2, #action="extend",
    help="returns timezone of unaware timeseries")
parser.add_argument("-s", "--set_timezone", nargs=1,
    help="sets the timezone of an unaware timeseries"    
    )

if __name__=="__main__":
    args = parser.parse_args()
    if args.get_timezone:
        for arg in args.get_timezone:
            if not is_csv(arg):
                raise TypeError(f"File {arg} must be a '.csv' file")
        pass
        file_1 = get_filename(args.get_timezone[0])
        file_2 = get_filename(args.get_timezone[1])

        try:
            df1 = open_csv(file_1)
            df2 = open_csv(file_2)
        except ValueError:
            raise(
                "The file you supplied is in the wrong format. Are you sure "
                "you have a ohlc file?"
                )
        utc_timezone = get_idx(df1, df2)
        print(utc_timezone)
    if args.set_timezone:
        if not is_csv(args.set_timezone[0]):
            raise TypeError(f"File {arg} must be a '.csv' file")
        df = set_tz(args.set_timezone[0], args.set_timezone[1])
        df.to_csv(args.set_timezone[0])
        print(
            f"Timezone of {args.set_timezone[0]} set to {args.set_timezone[1]}"
            )
