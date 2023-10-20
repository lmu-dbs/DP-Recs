import os

import pandas as pd
import datetime

from utils import is_sorted, write_dataset, split_training_test


def import_helpdesk():
    df = pd.read_csv(r'datasets/helpdesk/helpdesk.csv')
    df = df.groupby('CaseID').agg({'ActivityID': lambda x: list(x), 'CompleteTimestamp': lambda x: list(x)})
    print(df)
    act = []
    timestamps = []
    not_sorted = 0
    for index, row in df.iterrows():
        act.append([int(x) for x in row["ActivityID"]])
        ts = [int((datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S") - datetime.datetime(1900, 1, 1)).total_seconds()) for time in row["CompleteTimestamp"]]
        if not is_sorted(ts):
            not_sorted += 1
        timestamps.append(ts)
    # print(not_sorted)
    # print(act)
    # print(timestamps)
    if not_sorted == 0:
        train, test = split_training_test(act, int(len(act)*0.7), 1)
        train_ts, test_ts = split_training_test(timestamps, int(len(act)*0.7), 1)
        write_dataset("helpdesk.txt", train, test, train_ts, test_ts)
    else:
        raise ImportError("Sequences not sorted")
    return df


def import_data(file):
    """Reads the "TrackingObjects.csv"-files and outputs them as a pandas dataframe

    Parameters
    ----------
    path : string
        path to the TrackinObjects-files

    filename : string
        the name of the trackings-objects file

    Returns
    -------
    Pandas.Dataframe
        The dataframe that will be used for analysis
        :param file:
    """

    df = pd.read_csv(os.path.join(file),
                     sep="~",
                     encoding='latin1',
                     header=None,
                     index_col=False,
                     parse_dates=[5],
                     names=[
                         "Id",
                         "BrowserEnvironmentId",
                         "EventParams",
                         "concept:name",
                         "LoggedIn",
                         "time:timestamp",
                         "Language",
                         "na1",
                         "case:concept:name",
                         "na2"
                     ],
                     dtype={"Id": 'int64',
                            "BrowserEnvironmentId": 'int64',
                            "concept:name": 'str',
                            "EventParams": "str",
                            "LoggedIn": 'int64',
                            "Language": 'str',
                            "case:concept:name": 'str'
                            }
                     )
    return df


if __name__ == "__main__":
    helpdesk_df = import_helpdesk()
    # print(helpdesk_df)


