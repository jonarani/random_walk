import enum
import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Create and algorithm for random walk with equal probability of length 100
# 2. Run your RW algorithm 10x and calculate the statistical features: 
#    time average, ensemble average, variance and standard deviation.
#    Plot a histogram of the ensemble, point out the mean and standard deviation.
#    Are your random walks stationary or non stationary, and why?

RW_LENGTH = 100
NR_OF_WALKS = 10
STEPS = np.array([-1, 1])

# Random walk array
X = np.zeros((NR_OF_WALKS, RW_LENGTH))

# Time-average mean value
M = np.zeros(1)
M2 = np.zeros((NR_OF_WALKS, 1))

# Ensemble-average
ensemble = np.zeros(RW_LENGTH)
# Ensemble mean
ensemble_mean = 0
# For calculating standard deviation
ensemble_tmp = np.zeros(RW_LENGTH)
ensemble_strd_dev = 0

# Variance temp array
var_tmp = np.zeros((NR_OF_WALKS, RW_LENGTH))
# Variances
vars = np.zeros((NR_OF_WALKS, 1))

# Standard deviations
strd_dev = np.zeros((NR_OF_WALKS, 1))

# Xi[0] is starting point number 0
for i in range (NR_OF_WALKS):
    for t in range(RW_LENGTH - 1):
        indx = random.randint(0, 1)
        # Xi[t] = Xi[t-1] + Ei[t]
        X[i][t + 1] = X[i][t] + STEPS[indx]
    
    # time-average mean
    M2[i] = np.mean(X[i])

    # Variance
    var_tmp[i] = np.subtract(X[i], M2[i])
    var_tmp[i] = var_tmp[i] * var_tmp[i]
    vars[i] = np.sum(var_tmp[i]) / RW_LENGTH

    # Standard deviation
    strd_dev[i] = np.sqrt(np.sum(var_tmp[i]) / (RW_LENGTH - 1))

M = np.mean(X) # Same as ensemble mean

print ("Means")
print (M2)
print ("Variances")
print (vars)
print ("Standard deviations")
print (strd_dev)

# Calculate ensemble
ensemble = np.sum(X, axis=0)
ensemble = ensemble / NR_OF_WALKS

# Ensemble time-average mean
ensemble_mean = np.mean(ensemble)

# Ensemble standard deviation
ensemble_tmp = np.subtract(ensemble, ensemble_mean)
ensemble_tmp = ensemble_tmp * ensemble_tmp
ensemble_strd_dev = np.sqrt(np.sum(ensemble_tmp) / (RW_LENGTH - 1))

print (ensemble)
print ("Ensemble mean: {}".format(ensemble_mean))
print ("Ensemble standard deviation: {}".format(ensemble_strd_dev))


flatX = X.flatten()
n, bins, patches = plt.hist(flatX, 25, facecolor='b')
plt.grid(True)
plt.show()

# 10 random walks
plt.plot(X[0], color='b', label='Random walks')
plt.plot(X[1], color='b')
plt.plot(X[2], color='b')
plt.plot(X[3], color='b')
plt.plot(X[4], color='b')
plt.plot(X[5], color='b')
plt.plot(X[6], color='b')
plt.plot(X[7], color='b')
plt.plot(X[8], color='b')
plt.plot(X[9], color='b')

# Time-average mean
plt.axhline(M, label='time-average', color='r', linestyle='-')

# Ensemble average
plt.plot(ensemble, color='g', label='Ensemble')


plt.ylabel("Y")
plt.xlabel("Number of steps")
plt.legend()
plt.show()

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2)

ax1.plot(X[0], label='random walk')
ax1.axhline(M2[0], label='time-average', color='r', linestyle='-')
ax1.set_title("RW1")

ax2.plot(X[1])
ax2.axhline(M2[1], label='time-average', color='r', linestyle='-')
ax2.set_title("RW2")

ax3.plot(X[2])
ax3.axhline(M2[2], label='time-average', color='r', linestyle='-')
ax3.set_title("RW3")

ax4.plot(X[3])
ax4.axhline(M2[3], label='time-average', color='r', linestyle='-')
ax4.set_title("RW4")

ax5.plot(X[4])
ax5.axhline(M2[4], label='time-average', color='r', linestyle='-')
ax5.set_title("RW5")

ax6.plot(X[5])
ax6.axhline(M2[5], label='time-average', color='r', linestyle='-')
ax6.set_title("RW6")

ax7.plot(X[6])
ax7.axhline(M2[6], label='time-average', color='r', linestyle='-')
ax7.set_title("RW7")

ax8.plot(X[7])
ax8.axhline(M2[7], label='time-average', color='r', linestyle='-')
ax8.set_title("RW8")

ax9.plot(X[8])
ax9.axhline(M2[8], label='time-average', color='r', linestyle='-')
ax9.set_title("RW9")

ax10.plot(X[9])
ax10.axhline(M2[9], label='time-average', color='r', linestyle='-')
ax10.set_title("RW10")

plt.legend()
plt.show()
