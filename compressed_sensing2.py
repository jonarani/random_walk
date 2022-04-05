# 3) Resample the 10 data SPL sets so that they have equal time spacing (e.g. remove duplicates, fill missing data with linear interpolation).
# 4) Redo 1) and 2). Compare the RMSE plots for both cases.
from calendar import c
from datetime import date
from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.ndimage
import datetime
import glob
from matplotlib.pyplot import figure
from scipy.interpolate import interp1d
import pandas as pd

DATA_DIR = 'tammsaare_street/*.csv'
first = '70B3D5E39000206E-data-2022-03-19 16 35 25.csv'
third = '70B3D5E39000237C-data-2022-03-19 16 33 03.csv'
tenth = '70B3D5E390002380-data-2022-03-19 16 34 35.csv'

DATES = {}
DATA = {}


def remove_duplicates(data, dates):
    res = [idx for idx, val in enumerate(dates) if val in dates[:idx]]   
    data = np.delete(data, res)
    dates = np.delete(dates, res)
    return data, dates

def get_datetime_diff(d1, d2):
    return ((d2 - d1).total_seconds() / 60)

def interpolate(data, dates):
    new_data = np.zeros(0)

    ts_range = pd.period_range(start=dates[0], end=dates[len(dates)-1], freq='T')
    new_dates = ts_range.to_timestamp()
    new_data = np.full(len(ts_range), np.nan)

    # TODO: optimize
    for i, new_date in enumerate(new_dates):
        for j, old_date in enumerate(dates):
            if (new_date == old_date):
                new_data[i] = data[j]
                break

    df = pd.DataFrame({'Date': new_dates, 'Value': new_data})
    res = df.set_index('Date')['Value'].interpolate(method="linear")

    return np.asarray(res.values, dtype=int), np.asarray(new_dates)

# Read data from CSV files to dictionaries
for i, file in enumerate(glob.glob(DATA_DIR)):
    with open(file, newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        decibels = np.zeros(0)
        dates = []

        for j, row in enumerate(csv_reader):
            date_str = row["\ufeffsep="]
            if (date_str != "Time"):
                date = datetime.datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]), int(date_str[11:13]), int(date_str[14:16]))
                dates = np.append(dates, date)

            val = row['']
            if (val != "dt_sound_level_dB"):
                val = int(val)
                decibels = np.append(decibels, val)

        # cut 'tammsaare_street' from the key
        key = file[17:]

        decibels = np.array(decibels, dtype=int)
        DATES[key] = dates
        DATA[key] = decibels

# Remove duplicates, interpolate
for key in DATA.keys():
    print ("Processing {}...".format(key))
    data, dates = remove_duplicates(DATA[key], DATES[key])
    data, dates = interpolate(data, dates)
    # Take the first 700
    DATES[key] = dates[:700]
    DATA[key] = data[:700]


# SVD
A = np.asarray((DATA['70B3D5E39000206E-data-2022-03-19 16 35 25.csv'], 
                DATA['70B3D5E39000235F-data-2022-03-19 16 33 37.csv'],
                DATA['70B3D5E39000237C-data-2022-03-19 16 33 03.csv'],
                DATA['70B3D5E390002007-data-2022-03-19 16 31 55.csv'],
                DATA['70B3D5E390002009-data-2022-03-19 16 28 17.csv'],
                DATA['70B3D5E390002021-data-2022-03-19 16 29 05.csv'],
                DATA['70B3D5E390002043-data-2022-03-19 16 30 39.csv'],
                DATA['70B3D5E390002047-data-2022-03-19 16 31 13.csv'],
                DATA['70B3D5E390002093-data-2022-03-19 16 30 01.csv'],
                DATA['70B3D5E390002380-data-2022-03-19 16 34 35.csv'],
                )) 

A = np.transpose(A)

#Performing SVD
U, D, VT = np.linalg.svd(A)

# 1st dimension denotes ranks
# 2nd dimension denotes the dataset
# 3rd dimension denotes sensors, where index 0 is dataset 0, index 1 dataset 2, index 2, dataset 9
A_remake = np.zeros((10, 700, 3))
realD = np.zeros((700, 10))
k = 10

for i in range(k):
    realD[i][i] = D[i]
    a_remake_k = U @ realD @ VT
    
    # Recreate 1st, 3rd and 10th dataset
    for c, d in zip([0, 2, 9], [0, 1, 2]):
        for r in range (700):
            A_remake[i][r][d] = a_remake_k[r][c]


# sensors, ranks
RMSE = np.zeros((3, 10))

for j in range (10): # ranks
    rmse1 = 0
    rmse2 = 0
    rmse3 = 0
    for k in range (700): # data
        rmse1 = rmse1 + DATA[first][k] - A_remake[j][k][0]
        rmse2 = rmse2 + DATA[third][k] - A_remake[j][k][1]
        rmse3 = rmse3 + DATA[tenth][k] - A_remake[j][k][2]

    rmse1 = np.sqrt(np.square(rmse1) / 700)
    rmse2 = np.sqrt(np.square(rmse2) / 700)
    rmse3 = np.sqrt(np.square(rmse3) / 700)

    RMSE[0][j] = rmse1
    RMSE[1][j] = rmse2
    RMSE[2][j] = rmse3

print (RMSE)

# dataset 1, 3, 10
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('ranks')
ax1.set_ylabel('Singular values', color=color)
ax1.plot(D, 'ro-', label='singular values')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('RMSE', color=color)  # we already handled the x-label with ax1
ax2.plot(RMSE[0], 'b.-', label='dataset1 rmse')
ax2.plot(RMSE[1], 'g.-', label='dataset3 rmse')
ax2.plot(RMSE[2], 'k.-', label='dataset10 rmse')
ax2.tick_params(axis='y', labelcolor=color)

plt.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
