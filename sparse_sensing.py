# 1) Using your RW script from week 7, create a downsampled, random and event-based sparse sampling of 10 RW 
#    (RW_01 - RW_10) with length 100.  Please make sure you use a continuous, uniform distribution for your 
#    RW this week. This means the values should be in between -1, +1 but not only these values.

# https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html

# "low" = -1
# "high" = 1

# 1a) Calculate a reference RW_ref as the ensemble average of the 10 RW. The RW_ref will be used in the following steps as your "ground truth" to compare all other data with.
# 1b) Downsample RW_01 by every 2nd value, and every 10th value.
# 1c) Randomly (uniform distribution) resample RW_01 with 10 and 50 samples.
# 1d) Perform event-based resampling of RW_01 with the thresholds of λ = 0.5 and 0.8. Don't forget to take the first derivative before thresholding!

import numpy as np
import matplotlib.pyplot as plt

# Parameters
RW_LENGTH = 100
NR_OF_WALKS = 10
HIGH = 1.0
LOW = -1.0

# Random walk array
RAND_WALKS = np.zeros((NR_OF_WALKS, RW_LENGTH))

# Ensemble average array
RW_ref = np.zeros(RW_LENGTH)


# Calculations
for i in range (NR_OF_WALKS):
    for t in range(RW_LENGTH - 1):
        # 1)
        RAND_WALKS[i][t + 1] = RAND_WALKS[i][t] + np.random.uniform(low=LOW, high=HIGH)

# 1a)
RW_ref = np.sum(RAND_WALKS, axis=0) / NR_OF_WALKS

# 1b)
every2nd_indices = np.arange(0, RAND_WALKS[0].size, 2)
downsampled_2nd_RW01 = np.delete(RAND_WALKS[0], every2nd_indices)

every10th_indices = np.arange(0, RAND_WALKS[0].size, 10)
downsampled_10th_RW01 = np.delete(RAND_WALKS[0], every10th_indices)

# 1c)
indices_10 = np.sort(np.random.uniform(low=0, high=RW_LENGTH, size=10).astype(int))
resampled_with10_RW01 = np.take(RAND_WALKS[0], indices_10)

indices_50 = np.sort(np.random.uniform(low=0, high=RW_LENGTH, size=50).astype(int))
resampled_with50_RW01 = np.take(RAND_WALKS[0], indices_50)

# 1d)
LAMBDA_1 = 0.5
evnt_bsd_05 = np.zeros(0)
t_indices_05 = np.zeros(0)

LAMBDA_2 = 0.8
evnt_bsd_08 = np.zeros(0)
t_indices_08 = np.zeros(0)

prev_t = 0
for t, val in enumerate(RAND_WALKS[0]):
    if (t == 0):
        dx_dt = 0 
    else:
        dx_dt = (RAND_WALKS[0][t] - RAND_WALKS[0][t - 1]) / (t - prev_t)
        prev_t = t

    if (abs(dx_dt) >= LAMBDA_1):
        evnt_bsd_05 = np.append(evnt_bsd_05, dx_dt)
        t_indices_05 = np.append(t_indices_05, t)

    if (abs(dx_dt) >= LAMBDA_2):
        evnt_bsd_08 = np.append(evnt_bsd_08, dx_dt)
        t_indices_08 = np.append(t_indices_08, t)

t_indices_05 = np.asarray(t_indices_05, dtype=int)
t_indices_08 = np.asarray(t_indices_08, dtype=int)

# 2) Calculate the RMSE of your 1b, 1c and 1d against the RW_ref.
#    Which method worked the best, and why?

# RMSE 1b
kept_indices_2nd = np.arange(1, RW_LENGTH, 2)
kept_indices_10th = np.delete(np.arange(0, RAND_WALKS[0].size, 1), every10th_indices)

rmse1b_1 = 0
for t, val in enumerate(kept_indices_2nd):
    rmse1b_1 = rmse1b_1 + np.square((RW_ref[val] - downsampled_2nd_RW01[t]))
rmse1b_1 = np.sqrt(rmse1b_1 / len(kept_indices_2nd))
print ("RMSE 1b_1: {}".format(rmse1b_1))

rmse1b_2 = 0
for t, val in enumerate(kept_indices_10th):
    rmse1b_2 = rmse1b_2 + np.square((RW_ref[val] - downsampled_10th_RW01[t]))
rmse1b_2 = np.sqrt(rmse1b_2 / len(kept_indices_10th))
print ("RMSE 1b_2: {}".format(rmse1b_2))

# RMSE 1c
rmse1c_1 = 0
for t, val in enumerate(indices_10):
    rmse1c_1 = rmse1c_1 + np.square((RW_ref[val] - resampled_with10_RW01[t]))
rmse1c_1 = np.sqrt(rmse1c_1 / len(indices_10))
print ("RMSE 1c_1: {}".format(rmse1c_1))

rmse1c_2 = 0
for t, val in enumerate(indices_50):
    rmse1c_2 = rmse1c_2 + np.square((RW_ref[val] - resampled_with50_RW01[t]))
rmse1c_2 = np.sqrt(rmse1c_2 / len(indices_50))
print ("RMSE 1c_2: {}".format(rmse1c_2))

# RMSE 1d
rmse1d_1 = 0
for t, val in enumerate(t_indices_05):
    rmse1d_1 = rmse1d_1 + np.square((RW_ref[val] - evnt_bsd_05[t]))
rmse1d_1 = np.sqrt(rmse1d_1 / len(t_indices_05))
print ("RMSE 1d_1: {}".format(rmse1d_1))

rmse1d_2 = 0
for t, val in enumerate(t_indices_08):
    rmse1d_2 = rmse1d_2 + np.square((RW_ref[val] - evnt_bsd_08[t]))
rmse1d_2 = np.sqrt(rmse1d_2 / len(t_indices_08))
print ("RMSE 1d_2: {}".format(rmse1d_2))


# Plotting
#plt.plot(RAND_WALKS[0], color='b', label='RW_01')
plt.plot(RW_ref, color='r', label='Ensemble')

plt.plot(indices_10, resampled_with10_RW01, 'k.--', label='resampled_10')
plt.plot(indices_50, resampled_with50_RW01, 'c.--', label='resampled_50')

plt.plot(t_indices_05, evnt_bsd_05, 'g:.', label='evnt-bsd resample 0.5')
plt.plot(t_indices_08, evnt_bsd_08, 'y:.', label='evnt-bsd resample 0.8')

plt.ylabel("Y")
plt.xlabel("Number of steps")
plt.legend()
plt.show()
