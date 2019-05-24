import numpy as np
import matplotlib.pyplot as plt
import math

x1 = np.arange(3, 4.6, 0.1)
y1 = np.full(len(x1), 0.4)

x2, y2 =[],[]

for _x in np.linspace(-180,180,360):
    x2.append(math.sin(math.radians(_x))+4.5)
    y2.append(math.cos(math.radians(_x))+1.4)

x3 = np.arange(4.5, 6, 0.1)
y3 = np.full(len(x3), 0.4)

x = np.append(np.append(x1, x2), x3)
y = np.append(np.append(y1, y2), y3)

plt.plot(x,y)
plt.show()