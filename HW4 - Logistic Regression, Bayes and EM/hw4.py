import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def sigmoid(self, X):
        w_x = -np.matmul(self.theta, X.T)
        return 1 / (1 + np.exp(w_x))

    def cost(self, X, y):
        sigmoid = self.sigmoid(X)
        h1 = np.matmul(-y, np.log(sigmoid))
        h0 = np.matmul((1 - y), np.log(1 - sigmoid))
        cost = (h1 - h0) / X.shape[0]
        return cost


    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        X = np.column_stack((np.ones(X.shape[0]), X))
        self.theta = np.random.rand(X.shape[1])
        for i in range(self.n_iter):
            curr_cost = self.cost(X, y)
            self.Js.append(curr_cost)

            if i > 0 and (self.Js[i - 1] - self.Js[i] < self.eps):
                break

            sigm = self.sigmoid(X)
            dw = np.dot((sigm - y).T, X) * (-self.eta)
            self.theta += dw

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        X = np.column_stack((np.ones(X.shape[0]), X))
        preds = np.where(self.sigmoid(X) > 0.5, 1, 0)
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    cv_accuracy = None
    np.random.seed(random_state)

    X = X.copy()
    y = y.copy()

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    fold_sizes = np.full(folds, X.shape[0] // folds, dtype=int)
    fold_sizes[:X.shape[0] % folds] += 1
    fold_indices = np.cumsum(fold_sizes)
    fold_indices = np.concatenate([[0], fold_indices])

    accuracies = []
    for i in range(folds):
        train_indices = np.concatenate([range(fold_indices[j], fold_indices[j + 1]) for j in range(folds) if j != i])
        valid_indices = range(fold_indices[i], fold_indices[i + 1])

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[valid_indices], y[valid_indices]

        algo.fit(X_train, y_train)
        preds = algo.predict(X_val)

        accuracy = np.mean(preds == y_val)
        accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    power = -0.5 * ((data - mu) / sigma) ** 2
    const = 1 / (sigma * np.sqrt(2 * np.pi))

    ans = np.exp(power)
    p = (const * ans)
    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.responsibilities = np.empty((self.k, data.shape[0]))
        min_val = np.min(data)
        max_val = np.max(data)
        self.mus = np.linspace(min_val, max_val, self.k)
        self.sigmas = np.ones(self.k)
        self.weights = np.full(self.k, 1 / self.k)
        self.costs = []


    def cost(self, data):
        w_pdf = self.weights[:, np.newaxis] * norm_pdf(data, self.mus[:, np.newaxis], self.sigmas[:, np.newaxis])
        log = -np.log(w_pdf)
        cost = np.sum(log, axis=0)
        return np.sum(cost)

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        p = self.weights[:, np.newaxis] * norm_pdf(data, self.mus[:, np.newaxis], self.sigmas[:, np.newaxis])
        sum_p_array = np.sum(p, axis=0)
        self.responsibilities = p / sum_p_array


    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        for i in range(self.k):
            self.weights[i] = np.mean(self.responsibilities[i])
            self.mus[i] = np.matmul(self.responsibilities[i], data.T) / (data.shape[0] * self.weights[i])
            mu_diff = data - self.mus[i]
            squared_diff = np.power(mu_diff, 2)
            sum_diff = np.matmul(self.responsibilities[i], squared_diff.T)
            self.sigmas[i] = np.sqrt(sum_diff / (data.shape[0] * self.weights[i]))

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """

        data = np.squeeze(data).copy()
        self.init_params(data)
        for i in range(self.n_iter):
            current_cost = self.cost(data)
            self.costs.append(current_cost)
            if i > 0 and np.abs(self.costs[i - 1] - self.costs[i]) <= self.eps:
                break
            self.expectation(data)
            self.maximization(data)

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = np.sum([weights[j] * norm_pdf(data, mus[j], sigmas[j]) for j in range(len(weights))], axis=0)
    return pdf
    #k = len(weights)
    #gmm_pdf = np.zeros_like(data)

    #for i in range(k):
     #   gmm_pdf += weights[i] * norm_pdf(data, mus[i], sigmas[i])
    #return gmm_pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.targets = None
        self.priors = {}
        self.models = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """

        self.targets = np.unique(y)
        self.models = np.empty((X.shape[1], len(self.targets)), dtype=EM)

        for target in self.targets:
            prior = (y == target).sum() / len(y)
            self.priors[target] = prior
            for f in range(X.shape[1]):
                model = EM(k=self.k)
                model.fit(X[y == target][:, f])
                self.models[f, target] = model

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None

        posteriors = np.zeros(X.shape[0])
        preds = np.zeros(X.shape[0])
        for target_idx, target in enumerate(self.targets):
            likelihoods = np.ones(X.shape[0])
            for f in range(X.shape[1]):
                model = self.models[f, target]
                weights, mus, sigmas = model.get_dist_params()
                probs = gmm_pdf(X[:, f], weights, mus, sigmas)

                likelihoods = likelihoods * probs

            new_posteriors = self.priors[target] * likelihoods
            preds[new_posteriors > posteriors] = target
            posteriors = np.maximum(posteriors, new_posteriors)
        return preds


# copied from the jupyter notebook
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    lor_train_acc = np.mean(lor.predict(x_train) == y_train)
    lor_test_acc = np.mean(lor.predict(x_test) == y_test)
    print(f'LOR accuracy for the training set: {lor_train_acc}')
    print(f'LOR accuracy for the test set: {lor_test_acc}')

    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)
    bayes_train_acc = np.mean(naive_bayes.predict(x_train) == y_train)
    bayes_test_acc = np.mean(naive_bayes.predict(x_test) == y_test)
    print(f'NaiveBayes accuracy for the training set: {bayes_train_acc}')
    print(f'NaiveBayes accuracy for the test set: {bayes_test_acc}')

    print(f'plot Small set - Logistic Regression {x_train.shape} {y_train.shape}')
    plt.title("Small set - Logistic Regression")
    plot_decision_regions(x_train, y_train, lor)

    print(f'plot Small set - Naive Bayes')
    plt.title("Small set - Naive Bayes")
    plot_decision_regions(x_train, y_train, naive_bayes)

    # print(f'lor: {lor.Js}')
    plt.title("Cost VS the Iteration - Logistic Regression")
    cost_1000 = plt.plot(np.arange(len(lor.Js)), lor.Js);
    plt.xscale('log');

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    from scipy.stats import multivariate_normal
    import numpy as np
    import matplotlib.pyplot as plt

    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    def generate_dataset_a():
        N = 500
        X = multivariate_normal.rvs(mean=[-2, -2, -2], cov=1, size=N)
        X = np.r_[X, multivariate_normal.rvs(mean=[0, 0, 0], cov=1, size=N)]
        X = np.r_[X, multivariate_normal.rvs(mean=[2, 2, 2], cov=1, size=N)]

        cl = np.r_[np.ones(N, dtype='int8'),
                   np.zeros(N, dtype='int8'),
                   np.ones(N, dtype='int8')
                   ]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.array(['green', 'blue'])[cl])

        return X, cl

    def generate_dataset_b():
        N = 500
        X = multivariate_normal.rvs(mean=[0, 0, -2], cov=[[2, 3, 3], [3, 2, 3], [3, 3, 2]], size=N)
        X = np.r_[X, multivariate_normal.rvs(mean=[0, 0, 2], cov=[[2, 3, 3], [3, 2, 3], [3, 3, 2]], size=N)]

        cl = np.r_[np.ones(N, dtype='int8'),
                   np.zeros(N, dtype='int8')
                   ]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.array(['green', 'blue'])[cl])

        return X, cl

    dataset_a_features, dataset_a_labels = generate_dataset_a()
    dataset_b_features, dataset_b_labels = generate_dataset_b()

    return {
        'dataset_a': {'features': dataset_a_features, 'labels': dataset_a_labels},
        'dataset_b': {'features': dataset_b_features, 'labels': dataset_b_labels}
    }

