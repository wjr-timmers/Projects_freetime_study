import pandas as pd
import time
import datetime
import numpy as np
import random


class Schedule:
    def __init__(self, begin_date, end_date):
        self.bd = begin_date
        self.ed = end_date

    def generate_empty(self):
        df = pd.DataFrame({"Date": pd.date_range(self.bd, self.ed)})
        df["Day"] = df.Date.dt.day_name()
        df["Week"] = df.Date.dt.isocalendar().week
        df["Year"] = df.Date.dt.year
        df["WN-Balie"] = 'None';
        df["HG-Balie1"] = 'None'
        df["HG-Balie2"] = 'None';
        df["OCC"] = 'None'
        df["CSC"] = 'None';
        df["SD1"] = 'None'
        df["SD2"] = 'None';
        df["SD3"] = 'None'
        df["SD4"] = 'None';
        df["SD5"] = 'None'
        df["SD6"] = 'None';
        df["Rest_Whole_day"] = 'None'
        df["Rest_Morning"] = 'None';
        df["Rest_Afternoon"] = 'None'
        df = df[df.Day != 'Saturday']
        df = df[df.Day != 'Sunday']

        indexNames1 = df[df['Day'] == 'Saturday'].index
        df.drop(indexNames1, inplace=True)
        indexNames2 = df[df['Day'] == 'Sunday'].index
        df.drop(indexNames2, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['Date'] = pd.to_datetime(df["Date"].dt.strftime('%Y-%m-%d'))
        return df

    def check_duplicates(self, listOfElems):
        while 'None' in listOfElems: listOfElems.remove('None')
        if len(listOfElems) == len(set(listOfElems)):
            return False
        else:
            return True

    def remove_choice(self, choice, whole_day, morning_day, afternoon_day):
        while choice in whole_day: whole_day.remove(choice)
        while choice in morning_day: morning_day.remove(choice)
        while choice in afternoon_day: afternoon_day.remove(choice)
        return whole_day, morning_day, afternoon_day

    def generate_random(self, schedule, availability):
        df = schedule.copy()

        for idx in range(len(df)):
            date = df.at[idx, "Date"]
            date_avail = availability[availability['Date'] == date]
            whole_day_list = date_avail['Whole Day A'].values
            whole_day = list(whole_day_list[0])
            morning_day_list = date_avail['Morning A'].values
            morning_day = list(morning_day_list[0])
            afternoon_day_list = date_avail['Afternoon A'].values
            afternoon_day = list(afternoon_day_list[0])

            choice = np.random.choice(np.array(whole_day))
            whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day, afternoon_day)
            df.at[idx, "WN-Balie"] = choice

            choice = np.random.choice(np.unique(np.append(np.array(morning_day), np.array(whole_day))))
            whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day, afternoon_day)
            df.at[idx, "HG-Balie1"] = choice

            choice = np.random.choice(np.unique(np.append(np.array(afternoon_day), np.array(whole_day))))
            whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day, afternoon_day)
            df.at[idx, "HG-Balie2"] = choice;

            choice = np.random.choice(np.array(whole_day))
            whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day, afternoon_day)
            df.at[idx, "CSC"] = choice

            if len(whole_day) != 0:
                choice = np.random.choice(np.array(whole_day))
                whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day,
                                                                           afternoon_day)
                df.at[idx, "SD1"] = choice

            if len(whole_day) != 0:
                choice = np.random.choice(np.array(whole_day))
                whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day,
                                                                           afternoon_day)
                df.at[idx, "SD2"] = choice

            if len(whole_day) != 0:
                choice = np.random.choice(np.array(whole_day))
                whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day,
                                                                           afternoon_day)
                df.at[idx, "SD3"] = choice

            if len(whole_day) != 0:
                choice = np.random.choice(np.array(whole_day))
                whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day,
                                                                           afternoon_day)
                df.at[idx, "SD4"] = choice

            if len(whole_day) != 0:
                choice = np.random.choice(np.array(whole_day))
                whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day,
                                                                           afternoon_day)
                df.at[idx, "SD5"] = choice

            if len(whole_day) != 0:
                choice = np.random.choice(np.array(whole_day))
                whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day,
                                                                           afternoon_day)
                df.at[idx, "SD6"] = choice

            if len(whole_day) != 0:
                choice = np.random.choice(np.array(whole_day))
                whole_day, morning_day, afternoon_day = self.remove_choice(choice, whole_day, morning_day,
                                                                           afternoon_day)
                df.at[idx, "OCC"] = choice

            df.at[idx, "Rest_Whole_day"] = list(dict.fromkeys(whole_day))
            df.at[idx, "Rest_Morning"] = list(dict.fromkeys(morning_day))
            df.at[idx, "Rest_Afternoon"] = list(dict.fromkeys(afternoon_day))
            plan = df.loc[
                idx, ['WN-Balie', 'HG-Balie1', 'HG-Balie2', 'OCC', 'CSC', 'SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6']]
            if self.check_duplicates(plan.tolist()) == True:
                raise ValueError('There is a double scheduling')
        return df