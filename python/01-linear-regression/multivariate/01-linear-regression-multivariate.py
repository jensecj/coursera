import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt # 2d plots
from mpl_toolkits.mplot3d import Axes3D # 3dplots
import numpy as np # matrices and numerical methods

data = np.loadtxt('ex1data2.txt', delimiter=',')
X = np.c_[data[:,0:2]]
y = np.c_[data[:,2]]

# this is a vectorized implementation, also works for multivarite
def computeCost(X, y, theta):
    m = X.shape[0] # number of examples

    predictions = X.dot(theta)
    errors = predictions - y
    sq_errors = np.square(errors)
    sum_of_sq_errors = np.sum(sq_errors)

    return (1/(2*m)) * sum_of_sq_errors

# note: we return the mean and standard deviation so we can normalize
# new test data we want to predict.
def normalizeFeatures(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_norm = X - mu
    X_norm = X_norm / sigma

    return (X_norm, mu, sigma)

X, mu, sigma = normalizeFeatures(X)
X = np.insert(X, 0, 1, axis=1) # add intercept term

theta = [[0],[0], [0]] # the parameters for our hypothesis
alpha = 0.05 # our learning rate for gradient descent
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
plt.xlim(0, 80)
plt.ylabel('Cost')
plt.xlabel('Iterations');
plt.savefig('01_gradient_descent_cost.png')
plt.close()

# when we add a new data point we want to predict, we need to
# normalize it first, before adding the intercept term
new_data = [1650, 3]
new_data = new_data - mu
new_data = new_data / sigma

# add intercept term
new_data = np.insert(new_data, 0, 1)

price = theta.T.dot(new_data)

print('Predicted price of a 1650 sq-ft, 3 br. house (using gradient descent): $%f' % price);
