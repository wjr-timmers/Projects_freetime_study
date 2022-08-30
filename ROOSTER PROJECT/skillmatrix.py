import pandas as pd
import time
import datetime
import numpy as np
import random

def get_matrix():
    df_skills = pd.read_csv('skillmatrix.CSV', sep=';',encoding = "ISO-8859-1")
    df_skills['VU ID'] = df_skills['VU ID'].str.replace(' ', '')

    # Remove coordinators
    df_skills = df_skills[df_skills['VU ID'] != 'MPL690']
    df_skills = df_skills[df_skills['VU ID'] != 'WTS250']

    all_persons = df_skills['VU ID'].unique()
    wn_persons = df_skills.loc[df_skills['WN Balie'] == 'x' , ['VU ID']]
    hg_persons = df_skills.loc[df_skills['HG Balie'] == 'x' , ['VU ID']]
    occ_persons = df_skills.loc[df_skills['OCC'] == 'x' , ['VU ID']]
    csc_persons = df_skills.loc[df_skills['CSC (uit giftes)'] == 'x' , ['VU ID']]
    sd_persons = df_skills.loc[df_skills['IT Servicedesk'] == 'x' , ['VU ID']]


    skills = {
        'wn': wn_persons.values.flatten(),
        'hg': hg_persons.values.flatten(),
        'occ': occ_persons.values.flatten(),
        'csc': csc_persons.values.flatten(),
        'sd': sd_persons.values.flatten()
    }

    #print(df_skills.head())
    return df_skills, skills, all_persons