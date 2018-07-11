import numpy as np
import matplotlib.pyplot as plot
import mpl_toolkits.mplot3d.axes3d


def fetch_data1():
    data1 = np.loadtxt("./ex1/ex1data1.csv", delimiter=",")
    x = data1[:, 0].reshape(1, -1)
    o = np.ones(x.size).reshape(1, -1)
    # (1, n)じゃないと転置で(n, 1)にできない
    x = np.vstack((o, x)).T
    y = data1[:, 1].reshape(1, -1).T

    return x, y


def warm_up_exercise():
    return np.identity(5)


def plot_data(x, y):
    plot.plot(x, y, 'rx', markersize=10)
    plot.xlabel('Population of City in 10,000s')
    plot.ylabel('Profit in $ 10,000s')


def compute_cost(x, y, theta):
    m = y.size
    costs = (x.dot(theta) - y) ** 2
    return costs.sum() / (2.0 * m)


def gradient_descent(x, y, theta, alpha, num_iters):
    m = y.size
    j_history = np.zeros(num_iters)

    for i in range(num_iters):
        h = x.dot(theta)
        errors = h - y
        delta = x.T.dot(errors)
        theta = theta - ((alpha / m) * delta)
        j_history[i] = compute_cost(x, y, theta)
        print(j_history[i])

    return theta, j_history


x, y = fetch_data1()
m = y.size
theta = np.zeros(x[0].size).reshape(1, -1).T
alpha = 0.01
num_iters = 1500
(theta, J_history) = gradient_descent(x, y, theta, alpha, num_iters)
predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))
print('Visualizing J(theta_0, theta_1) ...')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))
for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = compute_cost(x, y, t)
plot.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
fig = plot.figure()
ax = fig.add_subplot(111, projection='3d')
t0, t1 = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(t0, t1, J_vals)
plot.show()