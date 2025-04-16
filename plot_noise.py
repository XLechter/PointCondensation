import numpy as np
import matplotlib.pyplot as plt

x = [0, 100, 500, 2000, 4000]
y1 = [11.01, 36.34, 66.81, 72.62, 72.90]
y2 = [39.8, 53.00, 65.44, 72.01, 73.8]

plt.plot(x, y1, label='Gaussian')
plt.plot(x, y2, label='Random')
plt.xlabel('Iteration')
plt.ylabel('Accuracy(%)')
plt.legend()
#plt.show()
plt.savefig('noise.png')
