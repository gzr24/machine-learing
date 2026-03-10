import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)

y = np.sin(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='y = sin(x)', color='blue', linewidth=2)

plt.title('Visualization Test: Sine Wave', fontsize=14)
plt.xlabel('x (radians)', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.text(1, 0.5, r'$y = \sin(x)$', fontsize=15, color='red')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()


plt.show()
