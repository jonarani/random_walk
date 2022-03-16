import enum
import numpy as np
import random
import matplotlib.pyplot as plt

# 3. Calculate the logistic map function for l = 0.5, 2.5, 3.5, 4
#    Xn+1 = l * Xn (1 - Xn)
# 4. Compare the same statistical features of your RW (2) and the four
#    different logistic map time series (3)

LENGTH = 100
l = np.asarray([0.5, 2.5, 3.5, 4])
l_len = len(l)

means = np.zeros((l_len, 1))
variances = np.zeros((l_len, 1))
strd_deviations = np.zeros((l_len, 1))

ensemble = np.zeros(LENGTH)
ensemble_mean = 0
ensemble_strd_dev = 0

# Temporary storage for calculations
tmp_arr = np.zeros((l_len, LENGTH))

X = np.zeros((l_len, LENGTH))

for i in range(l_len):
    # Initial value
    X[i][0] = 0.3
    for n in range(LENGTH - 1):
        X[i][n+1] = l[i] * X[i][n] * (1 - X[i][n])

    means[i] = np.mean(X[i])

    # Standard deviation
    tmp_arr[i] = np.subtract(X[i], means[i])
    tmp_arr[i] = tmp_arr[i] * tmp_arr[i]
    variances[i] = np.sum(tmp_arr[i]) / LENGTH

    strd_deviations[i] = np.sqrt(np.sum(tmp_arr[i]) / (LENGTH - 1))
    


print ("Means")
print (means)
print ("Variances")
print (variances)
print ("Standard deviations")
print (strd_deviations)


# Calculate ensemble
ensemble = np.sum(X, axis=0)
ensemble = ensemble / l_len

# Ensemble time-average mean
ensemble_mean = np.mean(ensemble)

# Ensemble standard deviation
ensemble_tmp = np.subtract(ensemble, ensemble_mean)
ensemble_tmp = ensemble_tmp * ensemble_tmp
ensemble_strd_dev = np.sqrt(np.sum(ensemble_tmp) / (LENGTH - 1))

print ("Ensemble mean: {}".format(ensemble_mean))
print ("Ensemble standard deviation: {}".format(ensemble_strd_dev))

# Logistic map functions
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(X[0])
ax1.axhline(means[0], label='time-average', color='r', linestyle='-')
ax1.set_title("l = {}".format(l[0]))
ax2.plot(X[1])
ax2.set_title("l = {}".format(l[1]))
ax2.axhline(means[1], label='time-average', color='r', linestyle='-')
ax3.plot(X[2])
ax3.set_title("l = {}".format(l[2]))
ax3.axhline(means[2], label='time-average', color='r', linestyle='-')
ax4.plot(X[3])
ax4.set_title("l = {}".format(l[3]))
ax4.axhline(means[3], label='time-average', color='r', linestyle='-')
plt.show()



# Ensemble
# Time-average mean
plt.axhline(ensemble_mean, label='time-average', color='r', linestyle='-')
# Ensemble average
plt.plot(ensemble, color='b', label='Ensemble')
plt.legend()
plt.show()

flatX = X.flatten()
n, bins, patches = plt.hist(flatX, 25, facecolor='b')
plt.grid(True)
plt.show()