import matplotlib.pyplot as plt # 2d plots
from mpl_toolkits.mplot3d import Axes3D # 3dplots
import numpy as np # matrices and numerical methods

data = np.loadtxt('ex1data1.txt', delimiter=',')
X = np.c_[data[:,0]] # in the univariate case we only have 1 input
y = np.c_[data[:,1]] # expected outputs

plt.scatter(X, y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(0, 25)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.savefig('01_scatter_plot.png')
plt.close()

# Add intercept term (a column of 1's to the left of the matrix)
X = np.insert(X, 0, 1, axis=1)

# this is a naive implementation for the univariate case
def computeCost_naive(X, y, theta):
    m = X.shape[0] # number of examples
    n = X.shape[1] # number of features

    cost = 0
    for i in range(m): # iterate over all examples
        # the values are stored in arrays, we need to grab them
        Xval = X[i, 1]
        yval = y[i][0]
        theta0 = theta[0][0]
        theta1 = theta[1][0]

        # calculate the cost for each example and accumulate
        hyp = theta0 + theta1 * Xval
        error = hyp - yval
        cost += error * error;

    return (1 / (2 * m)) * cost

# this is a vectorized implementation, also works for multivarite
def computeCost(X, y, theta):
    m = X.shape[0] # number of examples

    predictions = X.dot(theta)
    errors = predictions - y
    sq_errors = np.square(errors)
    sum_of_sq_errors = np.sum(sq_errors)

    return (1/(2*m)) * sum_of_sq_errors

# note: we're using a placeholder for theta here, not a trained hyp.
print("naive cost: " + str(computeCost_naive(X, y, [[0],[0]])))
print("cost: " + str(computeCost(X, y, [[0],[0]])))

theta = [[0],[0]] # the parameters for our hypothesis
alpha = 0.01 # our learning rate for gradient descent
iterations = 1500 # how many iterations we're running gradient descent

# this is a vectorized implementation, also works for multivariate
def gradientDescent(X, y, theta, alpha, iterations):
    m = np.size(X, 0)
    J_hist = np.zeros(iterations)

    for i in np.arange(iterations):
        predictions = X.dot(theta)
        errors = predictions - y

        par_deriv = (1/m) * (X.T.dot(errors))
        theta = theta - alpha * par_deriv

        J_hist[i] = computeCost(X, y, theta)

    return (theta, J_hist)

theta, cost = gradientDescent(X, y, theta, alpha, iterations)
print('theta: ', theta.ravel())

plt.plot(cost)
plt.ylabel('Cost')
plt.xlabel('Iterations');
plt.savefig('02_gradient_descent_cost.png')
plt.close()

xs = np.arange(0,25) # we want to check for values 0...24
ys = theta[0] + theta[1] * xs # our predictions

plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xs, ys, label='test')
plt.xlim(0,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');
plt.savefig('03_population_example.png')
plt.close()

print('For population = 35,000, we predict a profit of $%f' % (theta.T.dot([1, 3.5]) * 10000))
print('For population = 70,000, we predict a profit of $%f' % (theta.T.dot([1, 7]) * 10000))

# create the values we need for plotting our gradient descent
theta0_axis = np.linspace(-10, 10, 100)
theta1_axis = np.linspace(-1, 4, 100)

xs, ys = np.meshgrid(theta0_axis, theta1_axis, indexing='xy')

J_vals = np.zeros((theta0_axis.size, theta1_axis.size))

# Calculate cost values
for (i,j),_ in np.ndenumerate(J_vals):
    J_vals[i,j] = computeCost(X, y, theta=[[xs[i,j]], [ys[i,j]]])

# create the coutour plot
plt.xlabel(r'$\theta_{0}$')
plt.ylabel(r'$\theta_{1}$');
plt.contour(xs, ys, J_vals, np.logspace(-2, 3, 30))
plt.scatter(theta[0], theta[1], c='r', marker="x");
plt.savefig('04_contour_plot.png')
plt.close()

# create the 3d surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, J_vals, rstride=3, cstride=3, cmap=plt.cm.rainbow)
ax.set_xlabel(r'$\theta_{0}$')
ax.set_ylabel(r'$\theta_{1}$')
ax.set_zlabel('Cost')
ax.set_zlim(J_vals.min(), J_vals.max())
ax.view_init(elev=15, azim=230)
plt.savefig('05_3d_surface_plot.png')
plt.close()
