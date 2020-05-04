import matplotlib.pyplot as plt

fig, axs = plt.subplots()

axs.plot([1,2,3,4], [1.3, 4,1,1, 6], label='x')
axs.title('Random signals')
axs.set_xlabel('Lags')
axs.set_ylabel('Other')
plt.show()