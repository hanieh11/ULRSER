from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

n_neighbors = 20

def AVG_acc(X, y, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=False)
    average_acc = 0
    classifiers = [KNN_clf]#, linear_SVM, rbf_SVM, LR]
    classifiers_acc = {'KNN_clf': 0, 'linear_SVM': 0, 'rbf_SVM': 0, 'LR': 0}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_acc = 0
        for clf in classifiers:
            acc = clf(X_train, X_test, y_train, y_test)
            fold_acc += acc
            assert clf.__name__ in classifiers_acc.keys()
            classifiers_acc[clf.__name__] += acc / n_folds

        fold_acc = fold_acc / len(classifiers)
        average_acc += fold_acc
    return average_acc / n_folds, classifiers_acc


def KNN_clf(X_train, X_test, y_train, y_test):
    global n_neighbors
    cls = KNN(n_neighbors=n_neighbors)
    cls.fit(X_train, y_train)
    return cls.score(X_test, y_test)


def linear_SVM(X_train, X_test, y_train, y_test):
    cls = SVC(kernel='linear')
    cls.fit(X_train, y_train)
    return cls.score(X_test, y_test)


def rbf_SVM(X_train, X_test, y_train, y_test):
    cls = SVC(kernel='rbf', gamma='scale')
    cls.fit(X_train, y_train)
    return cls.score(X_test, y_test)


def LR(X_train, X_test, y_train, y_test):
    cls = LogisticRegression(multi_class='auto', solver='newton-cg')
    cls.fit(X_train, y_train)
    return cls.score(X_test, y_test)

