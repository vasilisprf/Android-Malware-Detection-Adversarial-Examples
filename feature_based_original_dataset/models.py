from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from joblib import dump
import timeit
import numpy as np
import os

# init models
GNB = GaussianNB()  # Gaussian Naive Bayes
MNB = MultinomialNB()  # Multinomial Naive Bayes
CNB = ComplementNB()  # Complement Naive Bayes
BNB = BernoulliNB()  # Bernouli Naive Bayes
DT = DecisionTreeClassifier(criterion='gini', max_features=None, splitter='best')  # Decision Tree
RF = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features='log2')  # Random Forest
KNN = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='minkowski')  # K Nearest Neighbors
LR = LogisticRegression(solver='lbfgs', C=2.0, fit_intercept=True, max_iter=100)  # Logistic Regression model
SVM = svm.SVC(kernel='linear', C=0.5,  gamma='scale', decision_function_shape='ovr')  # Support Vector Machines
# the parameters are found from the grid search procedure.

path = "models/"
if not os.path.exists(path):
    os.mkdir(path)

"""
Each class defines a model for our classification task, containing four methods, namely 
train_, evaluate_, test_, get_average_metrics.
In train_ methods we fit the models.
    :param features: train data
    :param labels: train labels
    :param labels: save the model if True
    :return: 
In evaluate_ methods we get the accuracy on random test sets   
    :param model: the classifier
    :param features: test data
    :param labels: test labels
    :return:  
Train and evaluate are used in train_random_subsampling.py and in train_models.py  
In test_ methods we evaluate our models 
    :param test_features: validation data
    :param test_labels: validation labes
    :return: 
In get_average_metrics methods we get the average performance of each model because we evaluate each model x times 
on unseen data.
    :param val_runs: times to evaluate a model
Test and average metrics methods are used in evaluate_models.py
"""


class GaussianNaiveBayes:
    # metrics for the evaluation stage: FNR, FPR, accuracy
    average_FNR = 0
    average_FPR = 0
    average_accuracy = 0
    scores = []  # list for the testing training sets

    @staticmethod
    def train_gaussian_naive_bayes_classifier(features, labels, save=False):
        print("\n\n--- Training", type(GNB).__name__, "---")
        start_time = timeit.default_timer()  # timer
        model = GNB.fit(features, labels)  # fit model on training set
        stop_time = timeit.default_timer()
        print(type(GNB).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        if save:  # if defined, save the model
            print("Saving model...")
            filename = path + "model_" + type(GNB).__name__ + ".joblib"
            dump(model, filename)
        return model

    def evaluate_gaussian_naive_bayes_classifier(self, model, features, labels):
        print("\n\n--- Evaluating", type(GNB).__name__, "---")
        score = model.score(features, labels)  # evaluate the model in training stage
        print("Accuracy:", score * 100)
        self.scores.append(score)
        return self.scores

    def test_gaussian_naive_bayes_classifier(self, model, test_features, test_labels):
        print(type(GNB).__name__, "predicting...")
        start_time = timeit.default_timer()
        predicted = model.predict(test_features)  # evaluate the model in evaluation stage
        confusion = confusion_matrix(test_labels, predicted)  # confusion matrix metrics
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        FNR = FN / float(FN + TP) * 100
        FPR = FP / float(FP + TN) * 100
        accuracy = (TP + TN) / float(TP + TN + FP + FN) * 100
        print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
        print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

        stop_time = timeit.default_timer()
        print(type(GNB).__name__, "prediction time: ", stop_time - start_time, "seconds\n\n")
        self.average_FNR += FNR
        self.average_FPR += FPR
        self.average_accuracy += accuracy

    def get_average_metrics(self, val_runs):
        self.average_FNR = self.average_FNR/val_runs
        self.average_FPR = self.average_FPR/val_runs
        self.average_accuracy = self.average_accuracy/val_runs
        print("Average Accuracy:", self.average_accuracy, "- Average FPR:", self.average_FPR,
              "- Average FNR:", self.average_FNR)


class MultinomialNaiveBayes:

    average_FNR = 0
    average_FPR = 0
    average_accuracy = 0
    scores = []

    @staticmethod
    def train_multi_naive_bayes_classifier(features, labels, save=False):
        print("\n\n--- Training", type(MNB).__name__, "---")
        start_time = timeit.default_timer()
        model = MNB.fit(features, labels)
        stop_time = timeit.default_timer()
        print(type(MNB).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        if save:
            print("Saving model...")
            filename = path + "model_" + type(MNB).__name__ + ".joblib"
            dump(model, filename)
        return model

    def evaluate_multi_naive_bayes_classifier(self, model, features, labels):
        print("\n\n--- Evaluating", type(MNB).__name__, "---")
        score = model.score(features, labels)
        print("Accuracy:", score * 100)
        self.scores.append(score)
        return self.scores

    def test_multi_naive_bayes_classifier(self, model, test_features, test_labels):
        print(type(MNB).__name__, "predicting...")
        start_time = timeit.default_timer()
        predicted = model.predict(test_features)
        confusion = confusion_matrix(test_labels, predicted)
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        FNR = FN / float(FN + TP) * 100
        FPR = FP / float(FP + TN) * 100
        accuracy = (TP + TN) / float(TP + TN + FP + FN) * 100
        print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
        print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

        stop_time = timeit.default_timer()
        print(type(MNB).__name__, "prediction time: ", stop_time - start_time, "seconds\n\n")
        self.average_FNR += FNR
        self.average_FPR += FPR
        self.average_accuracy += accuracy

    def get_average_metrics(self, val_runs):
        self.average_FNR = self.average_FNR / val_runs
        self.average_FPR = self.average_FPR / val_runs
        self.average_accuracy = self.average_accuracy / val_runs
        print("Average Accuracy:", self.average_accuracy, "- Average FPR:", self.average_FPR,
              "- Average FNR:", self.average_FNR)

    @staticmethod
    def train_incremental(features, labels):
        print("\n\n--- Training", type(MNB).__name__, "---")
        start_time = timeit.default_timer()
        model = MNB.partial_fit(features, labels, classes=np.unique(labels))
        stop_time = timeit.default_timer()
        print(type(MNB).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        return model


class ComplementNaiveBayes:

    average_FNR = 0
    average_FPR = 0
    average_accuracy = 0
    scores = []

    @staticmethod
    def train_complement_naive_bayes_classifier(features, labels, save=False):
        print("\n\n--- Training", type(CNB).__name__, "---")
        start_time = timeit.default_timer()
        model = CNB.fit(features, labels)
        stop_time = timeit.default_timer()
        print(type(CNB).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        if save:
            print("Saving model...")
            filename = path + "model_" + type(CNB).__name__ + ".joblib"
            dump(model, filename)
        return model

    def evaluate_complement_naive_bayes_classifier(self, model, features, labels):
        print("\n\n--- Evaluating", type(CNB).__name__, "---")
        score = model.score(features, labels)
        print("Accuracy:", score * 100)
        self.scores.append(score)
        return self.scores

    def test_complement_naive_bayes_classifier(self, model, test_features, test_labels):
        print(type(CNB).__name__, "predicting...")
        start_time = timeit.default_timer()
        predicted = model.predict(test_features)
        confusion = confusion_matrix(test_labels, predicted)
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        FNR = FN / float(FN + TP) * 100
        FPR = FP / float(FP + TN) * 100
        accuracy = (TP + TN) / float(TP + TN + FP + FN) * 100
        print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
        print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

        stop_time = timeit.default_timer()
        print(type(CNB).__name__, "prediction time: ", stop_time - start_time, "seconds\n\n")
        self.average_FNR += FNR
        self.average_FPR += FPR
        self.average_accuracy += accuracy

    def get_average_metrics(self, val_runs):
        self.average_FNR = self.average_FNR / val_runs
        self.average_FPR = self.average_FPR / val_runs
        self.average_accuracy = self.average_accuracy / val_runs
        print("Average Accuracy:", self.average_accuracy, "- Average FPR:", self.average_FPR,
              "- Average FNR:", self.average_FNR)

    @staticmethod
    def train_incremental(features, labels):
        print("\n\n--- Training", type(CNB).__name__, "---")
        start_time = timeit.default_timer()
        model = CNB.partial_fit(features, labels, classes=np.unique(labels))
        stop_time = timeit.default_timer()
        print(type(CNB).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        return model


class BernoulliNaiveBayes:

    average_FNR = 0
    average_FPR = 0
    average_accuracy = 0
    scores = []

    @staticmethod
    def train_bernoulli_naive_bayes_classifier(features, labels, save=False):
        print("\n\n--- Training", type(BNB).__name__, "---")
        start_time = timeit.default_timer()
        model = BNB.fit(features, labels)
        stop_time = timeit.default_timer()
        print(type(BNB).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        if save:
            print("Saving model...")
            filename = path + "model_" + type(BNB).__name__ + ".joblib"
            dump(model, filename)
        return model

    def evaluate_bernoulli_naive_bayes_classifier(self, model, features, labels):
        print("\n\n--- Evaluating", type(BNB).__name__, "---")
        score = model.score(features, labels)
        print("Accuracy:", score * 100)
        self.scores.append(score)
        return self.scores

    def test_bernoulli_naive_bayes_classifier(self, model, test_features, test_labels):
        print(type(BNB).__name__, "predicting...")
        start_time = timeit.default_timer()
        predicted = model.predict(test_features)
        confusion = confusion_matrix(test_labels, predicted)
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        FNR = FN / float(FN + TP) * 100
        FPR = FP / float(FP + TN) * 100
        accuracy = (TP + TN) / float(TP + TN + FP + FN) * 100
        print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
        print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

        stop_time = timeit.default_timer()
        print(type(BNB).__name__, "prediction time: ", stop_time - start_time, "seconds\n\n")
        self.average_FNR += FNR
        self.average_FPR += FPR
        self.average_accuracy += accuracy

    def get_average_metrics(self, val_runs):
        self.average_FNR = self.average_FNR / val_runs
        self.average_FPR = self.average_FPR / val_runs
        self.average_accuracy = self.average_accuracy / val_runs
        print("Average Accuracy:", self.average_accuracy, "- Average FPR:", self.average_FPR,
              "- Average FNR:", self.average_FNR)


class DecisionTree:

    average_FNR = 0
    average_FPR = 0
    average_accuracy = 0
    scores = []

    @staticmethod
    def train_decision_tree_classifier(features, labels, save=False):
        print("\n\n--- Training", type(DT).__name__, "---")
        start_time = timeit.default_timer()
        model = DT.fit(features, labels)
        stop_time = timeit.default_timer()
        print(type(DT).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        if save:
            print("Saving model...")
            filename = path + "model_" + type(DT).__name__ + ".joblib"
            dump(model, filename)
        return model

    def evaluate_decision_tree_classifier(self, model, features, labels):
        print("\n\n--- Evaluating", type(DT).__name__, "---")
        score = model.score(features, labels)
        print("Accuracy:", score * 100)
        self.scores.append(score)
        return self.scores

    def test_decision_tree_classifier(self, model, test_features, test_labels):
        print(type(DT).__name__, "predicting...")
        start_time = timeit.default_timer()
        predicted = model.predict(test_features)
        confusion = confusion_matrix(test_labels, predicted)
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        FNR = FN / float(FN + TP) * 100
        FPR = FP / float(FP + TN) * 100
        accuracy = (TP + TN) / float(TP + TN + FP + FN) * 100
        print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
        print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

        stop_time = timeit.default_timer()
        print(type(DT).__name__, "prediction time: ", stop_time - start_time, "seconds\n\n")
        self.average_FNR += FNR
        self.average_FPR += FPR
        self.average_accuracy += accuracy

    def get_average_metrics(self, val_runs):
        self.average_FNR = self.average_FNR / val_runs
        self.average_FPR = self.average_FPR / val_runs
        self.average_accuracy = self.average_accuracy / val_runs
        print("Average Accuracy:", self.average_accuracy, "- Average FPR:", self.average_FPR,
              "- Average FNR:", self.average_FNR)


class RandomForest:

    average_FNR = 0
    average_FPR = 0
    average_accuracy = 0
    scores = []

    @staticmethod
    def train_random_forest_classifier(features, labels, save=False):
        print("\n\n--- Training", type(RF).__name__, "---")
        start_time = timeit.default_timer()
        model = RF.fit(features, labels)
        stop_time = timeit.default_timer()
        print(type(RF).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        if save:
            print("Saving model...")
            filename = path + "model_" + type(RF).__name__ + ".joblib"
            dump(model, filename)
        return model

    def evaluate_random_forest_classifier(self, model, features, labels):
        print("\n\n--- Evaluating", type(RF).__name__, "---")
        score = model.score(features, labels)
        print("Accuracy:", score * 100)
        self.scores.append(score)
        return self.scores

    def test_random_forest_classifier(self, model, test_features, test_labels):
        print(type(RF).__name__, "predicting...")
        start_time = timeit.default_timer()
        predicted = model.predict(test_features)
        confusion = confusion_matrix(test_labels, predicted)
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        FNR = FN / float(FN + TP) * 100
        FPR = FP / float(FP + TN) * 100
        accuracy = (TP + TN) / float(TP + TN + FP + FN) * 100
        print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
        print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

        stop_time = timeit.default_timer()
        print(type(RF).__name__, "prediction time: ", stop_time - start_time, "seconds\n\n")
        self.average_FNR += FNR
        self.average_FPR += FPR
        self.average_accuracy += accuracy

    def get_average_metrics(self, val_runs):
        self.average_FNR = self.average_FNR / val_runs
        self.average_FPR = self.average_FPR / val_runs
        self.average_accuracy = self.average_accuracy / val_runs
        print("Average Accuracy:", self.average_accuracy, "- Average FPR:", self.average_FPR,
              "- Average FNR:", self.average_FNR)


class KNearestNeighbors:

    average_FNR = 0
    average_FPR = 0
    average_accuracy = 0
    scores = []

    @staticmethod
    def train_knn_classifier(features, labels, save=False):
        print("\n\n--- Training", type(KNN).__name__, "---")
        start_time = timeit.default_timer()
        model = KNN.fit(features, labels)
        stop_time = timeit.default_timer()
        print(type(KNN).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        if save:
            print("Saving model...")
            filename = path + "model_" + type(KNN).__name__ + ".joblib"
            dump(model, filename)
        return model

    def evaluate_knn_classifier(self, model, features, labels):
        print("\n\n--- Evaluating", type(KNN).__name__, "---")
        score = model.score(features, labels)
        print("Accuracy:", score * 100)
        self.scores.append(score)
        return self.scores

    def test_knn_classifier(self, model, test_features, test_labels):
        print(type(KNN).__name__, "predicting...")
        start_time = timeit.default_timer()
        predicted = model.predict(test_features)
        confusion = confusion_matrix(test_labels, predicted)
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        FNR = FN / float(FN + TP) * 100
        FPR = FP / float(FP + TN) * 100
        accuracy = (TP + TN) / float(TP + TN + FP + FN) * 100
        print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
        print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

        stop_time = timeit.default_timer()
        print(type(KNN).__name__, "prediction time: ", stop_time - start_time, "seconds\n\n")
        self.average_FNR += FNR
        self.average_FPR += FPR
        self.average_accuracy += accuracy

    def get_average_metrics(self, val_runs):
        self.average_FNR = self.average_FNR / val_runs
        self.average_FPR = self.average_FPR / val_runs
        self.average_accuracy = self.average_accuracy / val_runs
        print("Average Accuracy:", self.average_accuracy, "- Average FPR:", self.average_FPR,
              "- Average FNR:", self.average_FNR)


class LogRegression:

    average_FNR = 0
    average_FPR = 0
    average_accuracy = 0
    scores = []

    @staticmethod
    def train_logistic_regression_classifier(features, labels, save=False):
        print("\n\n--- Training", type(LR).__name__, "---")
        start_time = timeit.default_timer()
        model = LR.fit(features, labels)
        stop_time = timeit.default_timer()
        print(type(LR).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        if save:
            print("Saving model...")
            filename = path + "model_" + type(LR).__name__ + ".joblib"
            dump(model, filename)
        return model

    def evaluate_logistic_regression_classifier(self, model, features, labels):
        print("\n\n--- Evaluating", type(LR).__name__, "---")
        score = model.score(features, labels)
        print("Accuracy:", score * 100)
        self.scores.append(score)
        return self.scores

    def test_logistic_regression_classifier(self, model, test_features, test_labels):
        print(type(LR).__name__, "predicting...")
        start_time = timeit.default_timer()
        predicted = model.predict(test_features)
        confusion = confusion_matrix(test_labels, predicted)
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        FNR = FN / float(FN + TP) * 100
        FPR = FP / float(FP + TN) * 100
        accuracy = (TP + TN) / float(TP + TN + FP + FN) * 100
        print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
        print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

        stop_time = timeit.default_timer()
        print(type(LR).__name__, "prediction time: ", stop_time - start_time, "seconds\n\n")
        self.average_FNR += FNR
        self.average_FPR += FPR
        self.average_accuracy += accuracy

    def get_average_metrics(self, val_runs):
        self.average_FNR = self.average_FNR / val_runs
        self.average_FPR = self.average_FPR / val_runs
        self.average_accuracy = self.average_accuracy / val_runs
        print("Average Accuracy:", self.average_accuracy, "- Average FPR:", self.average_FPR,
              "- Average FNR:", self.average_FNR)


class SupportVectorMachine:

    average_FNR = 0
    average_FPR = 0
    average_accuracy = 0
    scores = []

    @staticmethod
    def train_svm_classifier(features, labels, save=False):
        print("\n\n--- Training", type(SVM).__name__, "---")
        start_time = timeit.default_timer()
        model = SVM.fit(features, labels)
        stop_time = timeit.default_timer()
        print(type(SVM).__name__, "training time: ", stop_time - start_time, "seconds\n\n")
        if save:
            print("Saving model...")
            filename = path + "model_" + type(SVM).__name__ + ".joblib"
            dump(model, filename)
        return model

    def evaluate_svm_classifier(self, model, features, labels):
        print("\n\n--- Evaluating", type(SVM).__name__, "---")
        score = model.score(features, labels)
        print("Accuracy:", score * 100)
        self.scores.append(score)
        return self.scores

    def test_svm_classifier(self, model, test_features, test_labels):
        print(type(SVM).__name__, "predicting...")
        start_time = timeit.default_timer()
        predicted = model.predict(test_features)
        confusion = confusion_matrix(test_labels, predicted)
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        FNR = FN / float(FN + TP) * 100
        FPR = FP / float(FP + TN) * 100
        accuracy = (TP + TN) / float(TP + TN + FP + FN) * 100
        print("FP:", FP, "- FN:", FN, "- TP:", TP, "- TN", TN)
        print("Accuracy:", accuracy, "- FPR:", FPR, "- FNR:", FNR)

        stop_time = timeit.default_timer()
        print(type(SVM).__name__, "prediction time: ", stop_time - start_time, "seconds\n\n")
        self.average_FNR += FNR
        self.average_FPR += FPR
        self.average_accuracy += accuracy

    def get_average_metrics(self, val_runs):
        self.average_FNR = self.average_FNR / val_runs
        self.average_FPR = self.average_FPR / val_runs
        self.average_accuracy = self.average_accuracy / val_runs
        print("Average Accuracy:", self.average_accuracy, "- Average FPR:", self.average_FPR,
              "- Average FNR:", self.average_FNR)
