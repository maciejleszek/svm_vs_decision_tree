import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class LinearSVMUsingSoftMargin:
    def __init__(self, C=1.0):
        self._support_vectors = None
        self.C = C
        self.beta = None
        self.b = None
        self.X = None
        self.y = None
        self.n = 0
        self.d = 0

    def __decision_function(self, X):
        return X.dot(self.beta) + self.b

    def __cost(self, margin):
        return (1 / 2) * self.beta.dot(self.beta) + self.C * np.sum(np.maximum(0, 1 - margin))

    def __margin(self, X, y):
        return y * self.__decision_function(X)

    def fit(self, X, y, lr=1e-3, epochs=500):
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.b = 0
        self.X = X
        self.y = y

        loss_array = []
        for _ in range(epochs):
            margin = self.__margin(X, y)
            misclassified_pts_idx = np.where(margin < 1)[0]
            d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            self.beta = self.beta - lr * d_beta
            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b
            loss = self.__cost(margin)
            loss_array.append(loss)

        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]

    def predict(self, X):
        return np.sign(self.__decision_function(X))

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(y == P)

    def plot_decision_boundary(self, X, y, lr_model=None, title=None):
        plt.figure(figsize=(10, 6))

        # Plot data points and decision boundary for SVM
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=50, alpha=0.7)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.__decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, colors=['r', 'b', 'r'], levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])

        # Plot support vectors if available
        if self._support_vectors is not None and len(self._support_vectors) > 0:
            sv_indices = self._support_vectors[self._support_vectors < len(X)]
            sv_x = X[sv_indices]  # Extract support vectors from X
            sv_y = y[sv_indices]  # Extract corresponding labels
            ax.scatter(sv_x[:, 0], sv_x[:, 1], s=100, linewidth=1, facecolors='none',
                       edgecolors='k')

        # Plot linear regression line if provided
        if lr_model is not None:
            coefficients = lr_model.coefficients[1:]  # Skip the intercept
            intercept = lr_model.coefficients[0]
            plt.plot(xx, -(intercept + coefficients[0] * xx) / coefficients[1], 'g--', linewidth=2,
                     label="Linear Regression")

        # Set labels, title, and legend
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(title)
        plt.legend()
        plt.colorbar()
        plt.show()


class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept (bias)
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Compute regression coefficients using the normal equation
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        # Add a column of ones to X for the intercept (bias)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X.dot(self.coefficients)

    def score(self, X, y):
        # Compute mean squared error (MSE) as a measure of model evaluation
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        return mse

class SVMDualProblem:
    def __init__(self, C=1.0, kernel='rbf', sigma=0.1, degree=1):
        self.C = C
        if kernel == 'poly':
            self.kernel = self._polynomial_kernel
            self.c = 1
            self.degree = degree
        else:
            self.kernel = self._rbf_kernel
            self.sigma = sigma

        self.X = None
        self.y = None
        self.alpha = None
        self.b = 0
        self.ones = None

    def _rbf_kernel(self, X1, X2):
        return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)

    def _polynomial_kernel(self, X1, X2):
        return (self.c + X1.dot(X2.T)) ** self.degree

    def fit(self, X, y, lr=1e-3, epochs=500):
        self.X = X
        self.y = y
        self.alpha = np.random.random(X.shape[0])
        self.b = 0
        self.ones = np.ones(X.shape[0])

        y_iy_jk_ij = np.outer(y, y) * self.kernel(X, X)

        for _ in range(epochs):
            gradient = self.ones - y_iy_jk_ij.dot(self.alpha)
            self.alpha = np.clip(self.alpha + lr * gradient, 0, self.C)

        index = np.where((self.alpha) > 0 & (self.alpha < self.C))[0]
        b_i = y[index] - (self.alpha * y).dot(self.kernel(X, X[index]))
        self.b = np.mean(b_i)

    def _decision_function(self, X):
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.b

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)


class SVMNonlinearRBF:
    def __init__(self, C, sigma):
        self.C = C
        self.sigma = sigma
        self.model = SVMDualProblem(C=self.C, kernel='rbf', sigma=self.sigma)

    def fit(self, X, y, lr=1e-3, epochs=500):
        self.model.fit(X, y, lr=lr, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class SVMNonlinearPoly:
    def __init__(self, C, degree=1):
        self.C = C
        self.degree = degree
        self.model = SVMDualProblem(C=self.C, kernel='poly', degree=self.degree)

    def fit(self, X, y, lr=1e-3, epochs=500):
        self.model.fit(X, y, lr=lr, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

def test_svm_kernels(X_train, X_test, y_train, y_test, C_values, sigma_values, degree_values):
    test_scores_rbf = {}
    test_scores_poly = {}

    # SVM with RBF Kernel
    for C in C_values:
        for sigma in sigma_values:
            svm_rbf = SVMNonlinearRBF(C=C, sigma=sigma)
            svm_rbf.fit(X_train, y_train, lr=1e-3, epochs=500)
            score_rbf = svm_rbf.score(X_test, y_test)
            test_scores_rbf[(C, sigma)] = score_rbf
            print(f"RBF: C={C}, sigma={sigma}, score={score_rbf}")

    # Find the best test score and corresponding parameters for RBF Kernel
    best_params_rbf = max(test_scores_rbf, key=test_scores_rbf.get)
    print(f"Best test score (RBF Kernel): {test_scores_rbf[best_params_rbf]} with params {best_params_rbf}")

    # SVM with Polynomial Kernel
    for C in C_values:
        for degree in degree_values:
            svm_poly = SVMNonlinearPoly(C=C, degree=degree)
            svm_poly.fit(X_train, y_train, lr=1e-3, epochs=500)
            score_poly = svm_poly.score(X_test, y_test)
            test_scores_poly[(C, degree)] = score_poly
            print(f"Poly: C={C}, degree={degree}, score={score_poly}")

    # Find the best test score and corresponding parameters for Polynomial Kernel
    best_params_poly = max(test_scores_poly, key=test_scores_poly.get)
    print(f"Best test score (Poly Kernel): {test_scores_poly[best_params_poly]} with params {best_params_poly}")

    return test_scores_rbf, test_scores_poly

def plot_kernel_scores(test_scores_rbf, test_scores_poly):
    # Plotting the test scores for SVM with RBF Kernel
    plt.figure(figsize=(8, 6))
    for (C, sigma), score in test_scores_rbf.items():
        plt.scatter(f"RBF: C={C}, sigma={sigma}", score, s=100, c='blue', alpha=0.5)
    plt.xticks(rotation=45)
    plt.title('Test Scores for SVM with RBF Kernel')
    plt.xlabel('Parameter Combination (C, sigma)')
    plt.ylabel('Test Score')
    plt.show()

    # Plotting the test scores for SVM with Polynomial Kernel
    plt.figure(figsize=(8, 6))
    for (C, degree), score in test_scores_poly.items():
        plt.scatter(f"Poly: C={C}, degree={degree}", score, s=100, c='green', alpha=0.5)
    plt.xticks(rotation=45)
    plt.title('Test Scores for SVM with Polynomial Kernel')
    plt.xlabel('Parameter Combination (C, degree)')
    plt.ylabel('Test Score')
    plt.show()

if __name__ == '__main__':
    # Specify the path to the wine dataset and the columns to use
    data_path = 'winequality-white.csv'
    cols = [0, 1]  # Example: Use first two columns as features

    # Load data and preprocess
    data = pd.read_csv(data_path, delimiter=';')
    X = data.drop(columns=['quality']).values
    y = data['quality'].values
    y[y <= 5] = -1
    y[y > 5] = 1

    if len(cols) > 0:
        X = X[:, cols]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear SVM
    # Initialize and train the Linear SVM model
    svm_model = LinearSVMUsingSoftMargin(C=15.0)
    svm_model.fit(X_train, y_train)

    # Evaluate the Linear SVM model
    svm_test_score = svm_model.score(X_test, y_test)
    print("Test score (Linear SVM):", svm_test_score)

    # Initialize and train the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Evaluate the Linear Regression model
    lr_test_score = lr_model.score(X_test, y_test)
    print("Test score (Linear Regression):", lr_test_score)

    # Plot decision boundary for both models
    svm_model.plot_decision_boundary(X_test, y_test, lr_model=lr_model, title="Linear SVM and Linear Regression Comparison")

    # Nonlinear SVM
    # Define lists of values to test for C and sigma
    C_values = [0.1, 1.0, 10.0]
    sigma_values = [0.1, 1.0, 10]
    degree_values = [1, 2, 3]

    # Test SVM kernels and find best parameters
    test_scores_rbf, test_scores_poly = test_svm_kernels(X_train, X_test, y_train, y_test, C_values, sigma_values, degree_values)

    # Plot test scores for SVM with RBF Kernel
    plot_kernel_scores(test_scores_rbf, test_scores_poly)

    print("Test score (Linear SVM):", svm_test_score)
    print("Test score (Linear Regression):", lr_test_score)

