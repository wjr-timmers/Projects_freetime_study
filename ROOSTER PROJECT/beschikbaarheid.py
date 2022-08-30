import pandas as pd
import time
import datetime
import numpy as np
import random

def get_beschikbaarheid():
    df_b = pd.read_csv('test.CSV')
    df_b['Subject'] = df_b['Subject'].str.upper()
    df_b['Subject'] = df_b['Subject'].str.replace(' ', '')
    df_b['Start Date'] = pd.to_datetime(df_b['Start Date'], dayfirst=True)
    df_b['End Date'] = pd.to_datetime(df_b['End Date'], dayfirst=True)

    # Change format
    df_b['Start Time'] = pd.to_datetime(df_b['Start Time'],format= '%H:%M:%S' ).dt.time
    df_b['End Time'] = pd.to_datetime(df_b['End Time'],format= '%H:%M:%S' ).dt.time
    df_b["Subject"] = df_b["Subject"].str.slice(0, 6, 1)

    # Remove coordinators
    df_b = df_b[df_b['Subject'] != 'MPL690']
    df_b = df_b[df_b['Subject'] != 'WTS250']

    #df_b.head()

    return df_b


def get_all_available_persons(all_persons):
    df_b = get_beschikbaarheid()
    person_avail = dict(zip(all_persons, [None] * len(all_persons)))

    for person in all_persons:
        person_avail[person] = df_b.loc[df_b['Subject'].str.contains(person)]

    return person_avail.keys()


def fine_tuning_availability(schedule, df_b):
    person_whole_day = []
    person_morning = []
    person_afternoon = []

    for idx in range(len(schedule)):
        date = schedule.loc[idx, "Date"]
        date_whole_day = df_b[(df_b['Start Date'] == date) & (df_b['All day event'] == True) |
                              (df_b['Start Date'] == date) & (df_b['Start Time'] <= datetime.time(8, 30, 0)) &
                              (df_b['End Time'] >= datetime.time(17, 0, 0))]
        date_vals = date_whole_day.copy()
        pers_avail = date_vals['Subject'].values
        person_whole_day.append(pers_avail)

        date_morning_day = df_b[(df_b['Start Date'] == date) & (df_b['All day event'] == True) |
                                (df_b['Start Date'] == date) & (df_b['Start Time'] <= datetime.time(8, 30, 0)) &
                                (df_b['End Time'] >= datetime.time(13, 0, 0))]
        date_vals2 = date_morning_day.copy()
        pers_avail_morning = date_vals2['Subject'].values
        person_morning.append(pers_avail_morning)

        date_afternoon_day = df_b[(df_b['Start Date'] == date) & (df_b['All day event'] == True) |
                                  (df_b['Start Date'] == date) & (df_b['Start Time'] <= datetime.time(12, 0, 0)) &
                                  (df_b['End Time'] >= datetime.time(17, 0, 0))]
        date_vals3 = date_afternoon_day.copy()
        pers_avail_afternoon = date_vals3['Subject'].values
        person_afternoon.append(pers_avail_afternoon)

    person_day_ser = pd.Series(person_whole_day)
    person_afternoon_ser = pd.Series(person_morning)
    person_morning_ser = pd.Series(person_afternoon)

    data_avail = [schedule["Date"], person_day_ser, person_afternoon_ser, person_morning_ser]

    headers = ["Date", "Whole Day A", "Morning A", "Afternoon A"]

    df_final_availability = pd.concat(data_avail, axis=1, keys=headers)
    #print(df_final_availability.at[0, "Whole Day A"])
    #print(df_final_availability.at[0, "Morning A"])
    #print(df_final_availability.at[0, "Afternoon A"])
    return df_final_availability