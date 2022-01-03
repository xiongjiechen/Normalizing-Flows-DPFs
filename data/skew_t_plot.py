import numpy as np
import matplotlib.pyplot as plt
from skewstudent import SkewStudent
from scipy.stats import norm

lam=0.8
x = np.arange(-3, 4, 0.01)
y1=SkewStudent(eta=3, lam=lam).pdf(x)
y2=SkewStudent(eta=2.1, lam=lam).pdf(x)
y3=SkewStudent(eta=30, lam=lam).pdf(x)
y4=norm.pdf(x)
plt.plot(x,y1, '-r')
plt.plot(x,y2, '.b')
plt.plot(x,y3, '*g')
plt.plot(x,y4, '.-c')

plt.show()
