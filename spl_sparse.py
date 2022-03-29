# 3)  Smooth the sound pressure files from (2) using a standard Gaussian filter (mean = 0, stddev = 1) with window sizes 11, 23, and 51 samples.
#     (Note: the 1D Gaussian kernel does not have to be perfectly symmetric, but it does not hurt if it is.)
# 3a) Downsample the SPL dataset by every 2nd value, and every 10th value.
# 3b) Randomly (uniform distribution) resample SPL dataset with 10, 50, 100 and 500 samples.
# 3c) Perform event-based resampling of the SPL dataset with the thresholds of λ = 5, 10 and 25.

from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.ndimage
import datetime
# 3)

WIN_SIZES = np.asarray([11, 23, 51])
SIGMA = 1

decibels = np.zeros(0)
dates = []
with open('spl_data.csv', newline='') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    for row in csv_reader:
        date_str = row["\ufeffTime"]
        date = datetime.datetime(int(date_str[6:10]), int(date_str[3:5]), int(date_str[0:2]), int(date_str[11:13]), int(date_str[14:16]))
        dates = np.append(dates, date)
        decibels = np.append(decibels, int(row["dt_sound_level_dB"]))


# https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
trun1 = ((WIN_SIZES[0] - 1 / 2) - 0.5) / SIGMA
trun2 = ((WIN_SIZES[1] - 1 / 2) - 0.5) / SIGMA
trun3 = ((WIN_SIZES[2] - 1 / 2) - 0.5) / SIGMA

gauss_filtered_1 = scipy.ndimage.gaussian_filter1d(decibels, SIGMA, truncate=trun1)
gauss_filtered_2 = scipy.ndimage.gaussian_filter1d(decibels, SIGMA, truncate=trun2)
gauss_filtered_3 = scipy.ndimage.gaussian_filter1d(decibels, SIGMA, truncate=trun3)

# 3a)
every2nd_indices = np.arange(0, decibels.size, 2)
downsampled_every_2nd = np.delete(decibels, every2nd_indices)

every10th_indices = np.arange(0, decibels.size, 10)
downsampled_every_10th = np.delete(decibels, every10th_indices)

# 3b)
indices_10 = np.sort(np.random.uniform(low=0, high=len(decibels), size=10).astype(int))
resampled_with10 = np.take(decibels, indices_10)

indices_50 = np.sort(np.random.uniform(low=0, high=len(decibels), size=50).astype(int))
resampled_with50 = np.take(decibels, indices_50)

indices_100 = np.sort(np.random.uniform(low=0, high=len(decibels), size=100).astype(int))
resampled_with100 = np.take(decibels, indices_100)

indices_500 = np.sort(np.random.uniform(low=0, high=len(decibels), size=500).astype(int))
resampled_with500 = np.take(decibels, indices_500)


# 3c)
LAMBDA_1 = 5
evnt_bsd_05 = np.zeros(0)
t_indices_05 = np.zeros(0)

LAMBDA_2 = 10
evnt_bsd_10 = np.zeros(0)
t_indices_10 = np.zeros(0)

LAMBDA_3 = 25
evnt_bsd_25 = np.zeros(0)
t_indices_25 = np.zeros(0)

for t, val in enumerate (decibels):
    if (t == 0):
        dx_dt = 0
    else:
        dt = (dates[t] - dates[t - 1]).total_seconds() / 60
        if (dt == 0):
            dx_dt = 0
        else:
            dx_dt = (decibels[t] - decibels[t - 1]) / dt

    if (abs(dx_dt) >= LAMBDA_1):
        evnt_bsd_05 = np.append(evnt_bsd_05, dx_dt)
        t_indices_05 = np.append(t_indices_05, t)

    if (abs(dx_dt) >= LAMBDA_2):
        evnt_bsd_10 = np.append(evnt_bsd_10, dx_dt)
        t_indices_10 = np.append(t_indices_10, t)
        
    if (abs(dx_dt) >= LAMBDA_3):
        evnt_bsd_25 = np.append(evnt_bsd_25, dx_dt)
        t_indices_25 = np.append(t_indices_25, t)

t_indices_05 = np.asarray(t_indices_05, dtype=int)
t_indices_10 = np.asarray(t_indices_10, dtype=int)
t_indices_25 = np.asarray(t_indices_25, dtype=int)


# 4) Calculate the RMSE of the original SPL dataset against 3a (10th value), 3b (10, 50 samples) and 3c (λ = 10).
#    What is the effect of smoothing on the sparse resampling of the SPL data?

kept_indices_10th = np.delete(np.arange(0, decibels.size, 1), every10th_indices)
rmse_3a = 0
for t, val in enumerate(kept_indices_10th):
    rmse_3a = rmse_3a + np.square((decibels[val] - downsampled_every_10th[t]))
    rmse_3a = np.sqrt(rmse_3a / len(kept_indices_10th))

print ("RMSE 3a: {}".format(rmse_3a))

rmse_3b_1 = 0
for t, val in enumerate(indices_10):
    rmse_3b_1 = rmse_3b_1 + np.square((decibels[val] - resampled_with10[t]))
    rmse_3b_1 = np.sqrt(rmse_3b_1 / len(indices_10))

print ("RMSE 3b_1: {}".format(rmse_3b_1))

rmse_3b_2 = 0
for t, val in enumerate(indices_50):
    rmse_3b_2 = rmse_3b_2 + np.square((decibels[val] - resampled_with50[t]))
    rmse_3b_2 = np.sqrt(rmse_3b_2 / len(indices_50))

print ("RMSE 3b_2: {}".format(rmse_3b_2))

rmse_3c = 0
for t, val in enumerate(t_indices_10):
    rmse_3c = rmse_3c + np.square((decibels[val] - evnt_bsd_10[t]))
    rmse_3c = np.sqrt(rmse_3c / len(t_indices_10))

print ("RMSE 3c: {}".format(rmse_3c))

# Plotting
plt.plot(decibels, 'b', label='spl')
#plt.plot(indices_10, resampled_with10, 'r--.', label='r10')
plt.plot(gauss_filtered_1, 'k', label='gauss 2')
# plt.plot(t_indices_05, evnt_bsd_05, 'k', label='evnt 5')
# plt.plot(t_indices_10, evnt_bsd_10, 'r', label='evnt 10')
# plt.plot(t_indices_25, evnt_bsd_25, 'c', label='evnt 25')

plt.ylabel("dB")
plt.xlabel("Samples")
plt.legend()
plt.show()

