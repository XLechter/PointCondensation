import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([0, 22, 91, 227, 454])*3
y1 = [39.8, 55.79, 65.13, 72, 73]
x2 = np.array([0, 46, 183, 460, 918])*3

y2 = [39.8, 53.24, 62, 63, 69]

plt.plot(x1, y1, label='Feature matching')
plt.plot(x2, y2, label='Gradient matching')
plt.xlabel('Time(s)')
plt.ylabel('Accuracy(%)')
plt.legend()
#plt.show()
plt.savefig('time.png')
