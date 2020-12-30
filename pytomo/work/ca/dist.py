import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def bi_triangle_cfd_inv(x, a, b):
    assert (x >= 0) and (x <= 1)
    aa = np.abs(a)
    h = 2. / (aa + b)
    if x < h*aa/4.:
        y = np.sqrt(x*aa/h) - aa
    elif x < h*aa/2.:
        y = -np.sqrt(aa*aa/2. - x*aa/h)
    elif x < (h*aa/2. + h*b/4.):
        y = np.sqrt(x*b/h - aa*b/2.)
    else:
        y = -np.sqrt(b/h * (1-x)) + b
    return y

def bi_triangle(a, b, rng):
    assert (a==b==0) or ((a < b) and (a <= 0) and (b >= 0))
    x_unif = rng.uniform(0, 1, 1)[0]
    x = bi_triangle_cfd_inv(x_unif, a, b)
    return x

a=-.5
b=1

rng = np.random.default_rng(0)

n=100000
ys = np.zeros(n, dtype='float')
for i in range(n):
    ys[i] = bi_triangle(a, b, rng)

sns.distplot(ys)
plt.show()
