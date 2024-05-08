import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class WineQualityClassifier:
    def __init__(self, data_path='winequality-white.csv'):
        # Wczytanie danych
        self.data = pd.read_csv(data_path, index_col=0, delimiter=';')
        self.X = self.data.drop(axis=1, labels=['quality'])
        self.y = self.data['quality'].values

        # Podział na zbiór testowy i treningowy
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0, stratify=self.y)

        # Inicjalizacja skalera cech
        self.scaler = StandardScaler()

    def train_svm(self):
        # Skalowanie cech dla SVM
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Parametry do GridSearchCV dla SVM
        svm_parameters = {
            'kernel': ['linear', 'rbf'],  # 2 rodzaje jader
            'C': [0.001, 0.01, 0.1, 1, 2, 5, 10],  # siła regularyzacji/karania modelu
            'gamma': [0.01, 0.1, 0.5, 1]
            # im wyższa wartość, tym model będzie bardziej wrażliwy na pojedyncze punkty
        }

        # Inicjalizacja modelu SVM i GridSearchCV
        self.svm_model = SVC(gamma='scale', random_state=8)
        self.svm_grid_search = GridSearchCV(self.svm_model, param_grid=svm_parameters, cv=3,
                                            n_jobs=-1, verbose=2)
        self.svm_grid_search.fit(self.X_train_scaled, self.y_train)

    def train_decision_tree(self):
        # Parametry do GridSearchCV dla drzewa decyzyjnego
        tree_parameters = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50]
        }

        # Inicjalizacja modelu drzewa decyzyjnego i GridSearchCV
        self.tree_model = DecisionTreeClassifier(random_state=8)
        self.tree_grid_search = GridSearchCV(self.tree_model, param_grid=tree_parameters, cv=3,
                                             n_jobs=-1, verbose=2)
        self.tree_grid_search.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        # Ewaluacja modeli - SVM
        best_svm_model = self.svm_grid_search.best_estimator_
        y_pred_svm = best_svm_model.predict(self.X_test_scaled)
        self.accuracy_svm = accuracy_score(self.y_test, y_pred_svm)

        # Ewaluacja modeli - drzewo decyzyjne
        best_tree_model = self.tree_grid_search.best_estimator_
        y_pred_tree = best_tree_model.predict(self.X_test)
        self.accuracy_tree = accuracy_score(self.y_test, y_pred_tree)

    def plot_mean_test_scores(self):
        # Wykresy Mean Test Scores - SVM (Linear i RBF Kernel)
        results_svm = self.svm_grid_search.cv_results_
        linear_scores_svm = [mean_score for mean_score, params in
                             zip(results_svm['mean_test_score'], results_svm['params']) if
                             params['kernel'] == 'linear']
        linear_param_combinations_svm = [(params['C'], params['gamma']) for params in
                                         results_svm['params'] if params['kernel'] == 'linear']
        linear_param_combinations_svm = np.array(linear_param_combinations_svm)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(linear_scores_svm)), linear_scores_svm, marker='o', linestyle='-',
                 label='Linear Kernel (SVM)')
        plt.xticks(range(len(linear_scores_svm)), linear_param_combinations_svm, rotation=45)

        rbf_scores_svm = [mean_score for mean_score, params in
                          zip(results_svm['mean_test_score'], results_svm['params']) if
                          params['kernel'] == 'rbf']
        rbf_param_combinations_svm = [(params['C'], params['gamma']) for params in
                                      results_svm['params'] if params['kernel'] == 'rbf']
        rbf_param_combinations_svm = np.array(rbf_param_combinations_svm)

        plt.plot(range(len(rbf_scores_svm)), rbf_scores_svm, marker='o', linestyle='-',
                 label='RBF Kernel (SVM)')
        plt.xticks(range(len(rbf_scores_svm)), rbf_param_combinations_svm, rotation=45)

        plt.title('Mean Test Scores - SVM')
        plt.xlabel('C, gamma')
        plt.ylabel('Mean Test Score')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Wykresy Mean Test Scores - Decision Tree (Gini i Entropy Criterion)
        results_tree = self.tree_grid_search.cv_results_
        linear_scores_tree = [mean_score for mean_score, params in
                              zip(results_tree['mean_test_score'], results_tree['params']) if
                              params['criterion'] == 'gini']
        linear_param_combinations_tree = [params['max_depth'] for params in results_tree['params']
                                          if params['criterion'] == 'gini']
        linear_param_combinations_tree = np.array(linear_param_combinations_tree)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(linear_scores_tree)), linear_scores_tree, marker='o', linestyle='-',
                 label='Gini Criterion (Decision Tree)')
        plt.xticks(range(len(linear_scores_tree)), linear_param_combinations_tree, rotation=45)

        rbf_scores_tree = [mean_score for mean_score, params in
                           zip(results_tree['mean_test_score'], results_tree['params']) if
                           params['criterion'] == 'entropy']
        rbf_param_combinations_tree = [params['max_depth'] for params in results_tree['params'] if
                                       params['criterion'] == 'entropy']
        rbf_param_combinations_tree = np.array(rbf_param_combinations_tree)

        plt.plot(range(len(rbf_scores_tree)), rbf_scores_tree, marker='o', linestyle='-',
                 label='Entropy Criterion (Decision Tree)')
        plt.xticks(range(len(rbf_scores_tree)), rbf_param_combinations_tree, rotation=45)

        plt.title('Mean Test Scores - Decision Tree')
        plt.xlabel('Max Depth')
        plt.ylabel('Mean Test Score')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def best_results(self):
        print("\nBest SVM Mean Test Score: {:.4f}".format(self.svm_grid_search.best_score_))
        print("Best Decision tree Mean Test Score: {:.4f}".format(self.tree_grid_search.best_score_))

        print("\nBest SVM params:", self.svm_grid_search.best_params_)
        print("Best Decision tree params:", self.tree_grid_search.best_params_)

        print("\nSVM accuracy:", self.accuracy_svm)
        print("Decision tree accuracy:", self.accuracy_tree)



wine_classifier = WineQualityClassifier()

wine_classifier.train_svm()
wine_classifier.train_decision_tree()

wine_classifier.evaluate_models()
wine_classifier.plot_mean_test_scores()
wine_classifier.best_results()
