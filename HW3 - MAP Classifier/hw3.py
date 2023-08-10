import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
          (0, 0): 0.1,
          (0, 1): 0.2,
          (1, 0): 0.2,
          (1, 1): 0.5
      }  # P(X=x, Y=y)

        self.X_C = {
          (0, 0): 0.1,
          (0, 1): 0.2,
          (1, 0): 0.4,
          (1, 1): 0.3
      }  # P(X=x, C=y)

        self.Y_C = {
          (0, 0): 0.1,
          (0, 1): 0.2,
          (1, 0): 0.4,
          (1, 1): 0.3
      }  # P(Y=y, C=c)

        self.X_Y_C = {
          (0, 0, 0): 0.02,
          (0, 0, 1): 0.08,
          (0, 1, 0): 0.08,
          (0, 1, 1): 0.12,
          (1, 0, 0): 0.08,
          (1, 0, 1): 0.12,
          (1, 1, 0): 0.32,
          (1, 1, 1): 0.18,
      }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        # Check if X and Y are not independent
        for x_value in X:
            for y_value in Y:
                if X_Y[(x_value, y_value)] != 0 and X_Y[(x_value, y_value)] != X[x_value] * Y[y_value]:
                    return True
        else:
            return False


    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        # P(X|C) and P(Y|C)
        p_x_given_c = {}
        p_y_given_c = {}
        for x in X:
            for c in C:
                p_x_given_c[(x, c)] = X_C.get((x, c), 0) / C.get(c, 1)
        for y in Y:
            for c in C:
                p_y_given_c[(y, c)] = Y_C.get((y, c), 0) / C.get(c, 1)

        # P(X,Y|C)
        p_x_y_given_c = {}
        for x in X:
            for y in Y:
                for c in C:
                    p_x_y_given_c[(x, y, c)] = X_Y_C.get((x, y, c), 0) / C.get(c, 1)

        # P(X,Y|C) as the product of P(X|C) and P(Y|C)
        p_x_y_given_c_expected = {}
        for x in X:
            for y in Y:
                for c in C:
                    p_x_y_given_c_expected[(x, y, c)] = p_x_given_c[(x, c)] * p_y_given_c[(y, c)]

        # Check if P(X,Y|C) matches the expected value
        for x in X:
            for y in Y:
                for c in C:
                    if abs(p_x_y_given_c[(x, y, c)] - p_x_y_given_c_expected[(x, y, c)]) > 1e-7:
                        return False
        return True



def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None

    rate = rate.copy()
    pos0 = np.where(rate == 0)
    rate2d_old = np.tile(rate, (k.shape[0], 1))
    rate[pos0] = 1
    rate2d = np.tile(rate, (k.shape[0], 1))
    pos2d = np.where(rate2d_old == 0)
    cum2d = np.tile(np.cumsum(np.log(np.arange(1, k.max()+1)))[k-1], (k.shape[0], 1))
    result = np.subtract(np.outer(k, np.log(rate)), rate2d) - cum2d.T
    result[pos2d] = -np.infty
    log_p = result

    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.sum(poisson_log_pmf(samples, rates), axis=0)

    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help

    return rates[np.argmax(get_poisson_log_likelihoods(samples, rates))]

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """

    return np.sum(samples)/len(samples)

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    power = -((x - mean) ** 2) / (2 * std ** 2)
    const = 1 / np.sqrt(2 * np.pi * std ** 2)
    p = (const * np.exp(power))
    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.value = class_value
        self.class_data = dataset[dataset[:, -1] == self.value][:, :-1]
        self.num_of_instances = len(dataset)
        self.mean = np.mean(self.class_data, axis=0)
        self.std = np.std(self.class_data, axis=0)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return len(self.class_data) / self.num_of_instances

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return np.prod(normal_pdf(x, self.mean, self.std))

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        if self.ccd1.get_instance_posterior(x) > self.ccd0.get_instance_posterior(x):
            return 1
        return 0

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = 0
    for x in test_set:
        if (map_classifier.predict(x[:-1]) == x[-1]):
            acc += 1
    return acc / len(test_set)



def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = np.power(np.power(np.sqrt(2 * np.pi), len(mean)) * np.linalg.det(cov), -0.5)  * np.power(np.e, (-(x - mean) @ np.linalg.inv(cov) @ (x - mean).T / 2))
    return pdf


class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.value = class_value
        self.class_data = dataset[dataset[:, -1] == self.value][:, :-1]
        self.num_of_instances = len(dataset)
        self.mean = np.mean(self.class_data, axis=0)
        self.cov = np.cov(self.class_data, rowvar=False)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return len(self.class_data) / self.num_of_instances

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov)

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        if self.ccd1.get_prior() > self.ccd0.get_prior():
            pred = 1
        else:
            pred = 0
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        likelihood_1 = self.ccd1.get_instance_likelihood(x)
        likelihood_0 = self.ccd0.get_instance_likelihood(x)

        if likelihood_1 is None or likelihood_0 is None:
            pred = 0  # or any default value
        else:
            if likelihood_1 > likelihood_0:
                pred = 1
            else:
                pred = 0
        return pred


EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


def vals_counted(data, feature=0, return_counts=True):
        vals, counts = np.unique(data[:,feature], return_counts=return_counts)
        return dict(zip(vals, counts))


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes
        distribution for a specific class. The probabilites are computed with laplace smoothing.

        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.value = class_value
        self.class_data = dataset[dataset[:, -1] == self.value][:, :-1]
        self.dataset = dataset
        self.num_of_instances = len(dataset)
        self.mean = np.mean(self.class_data, axis=0)
        self.std = np.std(self.class_data, axis=0)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return len(self.class_data) / self.num_of_instances

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        ni = len(self.class_data)
        p = 1

        for i in range(len(x)):
            Vj = len(np.unique(self.dataset[:, i]))
            nij = len(self.class_data[self.class_data[:, i] == x[i]])
            if nij == 0:
                p *= EPSILLON
            else:
                p *= (nij + 1) / (Vj + ni)

        return p

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        if self.ccd1.get_instance_posterior(x) > self.ccd0.get_instance_posterior(x):
            return 1
        return 0


    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None

        num_correct = 0
        for row in test_set:
            x = row[:-1]
            y = row[-1]
            y_hat = self.predict(x)
            if y_hat == y:
                num_correct += 1
        acc = num_correct / len(test_set)
        return acc

