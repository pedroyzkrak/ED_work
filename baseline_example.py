import time

import IPython.display as ipd
import numpy as np
import pandas as pd
import keras
from keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape
import math
from time import time as tm
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier


def knn_and_adaboost(neighbours,estimators,op="test"):
    print("Loading the files...")
    # pd.set_option('max_columns',20)
    tracks = pd.read_csv('tracks.csv', low_memory=False, skiprows=[1])
    features = pd.read_csv('features.csv',low_memory=False, skiprows=[1,2])

    columns_dict = {}
    for column in features.columns:
        if '.' in column and column.split('.')[0] in columns_dict.keys():
            columns_dict[column.split('.')[0]].append(column)
        else:
            columns_dict[column] = [column]

    subset = tracks.index[(tracks["set.1"] == 'medium') | (tracks["set.1"] == 'small')]

    tracks = tracks.loc[subset]
    features_all = features.loc[subset]

    print(tracks.shape, features_all.shape)

    train = tracks.index[tracks['set']  == 'training']
    val = tracks.index[tracks['set'] == 'validation']
    test = tracks.index[tracks['set'] == 'test']

    print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

    genres = list(LabelEncoder().fit(tracks['track.7']).classes_)

    print('Top genres ({}): {}'.format(len(genres), genres))


    def pre_process(tracks, features, columns, multi_label=False, verbose=False):
        if not multi_label:
            # Assign an integer value to each genre.
            enc = LabelEncoder()
            labels = tracks['track.7']
            # y = enc.fit_transform(tracks['track', 'genre_top'])
        else:
            # Create an indicator matrix.
            enc = MultiLabelBinarizer()
            labels = tracks['track.9']
            # labels = tracks['track', 'genres']

        # Split in training, validation and testing sets.
        #print("Spliting the dataset in training, validation and testing sets...")
        y_train = enc.fit_transform(labels[train])
        #print(y_train)
        y_val = enc.transform(labels[val])
        y_test = enc.transform(labels[test])
        X_train = (features.loc[train, columns]).as_matrix()
        #print(X_train)
        X_val = features.loc[val, columns].as_matrix()
        X_test = features.loc[test, columns].as_matrix()

        X_train, y_train = shuffle(X_train, y_train, random_state=42)




        # Standardize features by removing the mean and scaling to unit variance.
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_val)
        scaler.transform(X_test)

        return y_train, y_val, y_test, X_train, X_val, X_test

    def validate_classifiers_features(classifiers, feature_sets, multi_label=False):
        columns = list(classifiers.keys()).insert(0, 'dim')
        scores = pd.DataFrame(columns=columns, index=feature_sets.keys())
        for fset_name, fset in feature_sets.items():
            y_train, y_val, y_test, X_train, X_val, X_test = pre_process(tracks, features_all, fset, multi_label)
            scores.loc[fset_name, 'dim'] = X_train.shape[1]
            for clf_name, clf in classifiers.items():
                clf.fit(X_train, y_train)
                score = clf.score(X_val, y_val)
                scores.loc[fset_name, clf_name] = score
        return scores

    def test_classifiers_features(classifiers, feature_sets, multi_label=False):
        columns = list(classifiers.keys()).insert(0, 'dim')
        scores = pd.DataFrame(columns=columns, index=feature_sets.keys())
        times = pd.DataFrame(columns=classifiers.keys(), index=feature_sets.keys())
        start = tm()
        for fset_name, fset in feature_sets.items():
            start_comb = tm()
            y_train, y_val, y_test, X_train, X_val, X_test = pre_process(tracks, features_all, fset, multi_label)
            print("Combination: {}".format(fset_name))
            scores.loc[fset_name, 'dim'] = X_train.shape[1]
            for clf_name, clf in classifiers.items():
                start_classifier = tm()
                print("\tClassifier: {}".format(clf_name))
                t = time.process_time()
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores.loc[fset_name, clf_name] = score
                times.loc[fset_name, clf_name] = time.process_time() - t
                print("\tTime: {} s".format(tm() - start_classifier))
            print("Time for {}: {} s".format(fset_name, tm() - start_comb))
        print("Test Classifiers Features Finish.")
        print("Total time: {}".format(tm() - start))
        return scores, times

    classifiers = {
        'kNN': KNeighborsClassifier(n_neighbors=neighbours),
        'AdaBoost': AdaBoostClassifier(n_estimators=estimators),
    }


    feature_sets={
        'mfcc/contrast': columns_dict['mfcc']+columns_dict['spectral_contrast'],
        'mfcc/contrast/chroma': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['chroma_cens'],
        'mfcc/contrast/centroid': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['spectral_centroid'],
        'mfcc/contrast/chroma/centroid': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['chroma_cens']+columns_dict['spectral_centroid'],
        'mfcc/contrast/chroma/centroid/tonnetz': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['chroma_cens']+columns_dict['spectral_centroid']+columns_dict['tonnetz'],
        'mfcc/contrast/chroma/centroid/zcr': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['chroma_cens']+columns_dict['spectral_centroid']+columns_dict['zcr']
    }
    if op == 'validate':
        scores = validate_classifiers_features(classifiers, feature_sets)
        return scores

    scores, times = test_classifiers_features(classifiers, feature_sets)
    return scores, times

def baseline_example_all_classifiers():


    print("Loading the files...")
    #pd.set_option('max_columns',20)
    tracks = pd.read_csv('tracks.csv', low_memory=False, skiprows=[1])
    features = pd.read_csv('features.csv',low_memory=False, skiprows=[1,2])

    columns_dict = {}
    for column in features.columns:
        if '.' in column and column.split('.')[0] in columns_dict.keys():
            columns_dict[column.split('.')[0]].append(column)
        else:
            columns_dict[column] = [column]


    #np.testing.assert_array_equal(features.index, tracks.index)

    print(tracks.shape, features.shape)

    subset = tracks.index[(tracks["set.1"] == 'medium') | (tracks["set.1"] == 'small')]

    assert subset.isin(tracks.index).all()
    assert subset.isin(features.index).all()

    tracks = tracks.loc[subset]
    features_all = features.loc[subset]

    print(tracks.shape, features_all.shape)

    train = tracks.index[tracks['set']  == 'training']
    val = tracks.index[tracks['set'] == 'validation']
    test = tracks.index[tracks['set'] == 'test']

    print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

    genres = list(LabelEncoder().fit(tracks['track.7']).classes_)

    print('Top genres ({}): {}'.format(len(genres), genres))


    def pre_process(tracks, features, columns, multi_label=False, verbose=False):
        print("Pre-Process")
        if not multi_label:
            # Assign an integer value to each genre.
            enc = LabelEncoder()
            labels = tracks['track.7']
            # y = enc.fit_transform(tracks['track', 'genre_top'])
        else:
            # Create an indicator matrix.
            enc = MultiLabelBinarizer()
            labels = tracks['track.9']
            # labels = tracks['track', 'genres']

        # Split in training, validation and testing sets.
        print("Spliting the dataset in training, validation and testing sets...")
        y_train = enc.fit_transform(labels[train])
        #print(y_train)
        y_val = enc.transform(labels[val])
        y_test = enc.transform(labels[test])
        X_train = (features.loc[train, columns]).as_matrix()
        #print(X_train)
        X_val = features.loc[val, columns].as_matrix()
        X_test = features.loc[test, columns].as_matrix()

        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance.
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_val)
        scaler.transform(X_test)

        return y_train, y_val, y_test, X_train, X_val, X_test

    def test_classifiers_features(classifiers, feature_sets, multi_label=False):
        columns = list(classifiers.keys()).insert(0, 'dim')
        scores = pd.DataFrame(columns=columns, index=feature_sets.keys())
        times = pd.DataFrame(columns=classifiers.keys(), index=feature_sets.keys())
        start = tm()
        for fset_name, fset in feature_sets.items():
            start_comb = tm()
            y_train, y_val, y_test, X_train, X_val, X_test = pre_process(tracks, features_all, fset, multi_label)
            print("Combination: {}".format(fset_name))
            scores.loc[fset_name, 'dim'] = X_train.shape[1]
            for clf_name, clf in classifiers.items():
                start_classifier = tm()
                print("\tClassifier: {}".format(clf_name))
                t = time.process_time()
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores.loc[fset_name, clf_name] = score
                times.loc[fset_name, clf_name] = time.process_time() - t
                print("\tTime: {} s".format(tm() - start_classifier))
            print("Time for {}: {} s".format(fset_name, tm() - start_comb))
        print("Test Classifiers Features Finish.")
        print("Total time: {}".format(tm() - start))
        return scores, times

    classifiers = {
        'LR': LogisticRegression(),
        'kNN': KNeighborsClassifier(n_neighbors=200),
        'SVCrbf': SVC(kernel='rbf'),
        'SVCpoly1': SVC(kernel='poly', degree=1),
        'linSVC1': SVC(kernel="linear"),
        'linSVC2': LinearSVC(),
        'DT': DecisionTreeClassifier(max_depth=5),
        'RF': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        'AdaBoost': AdaBoostClassifier(n_estimators=10),
        'MLP1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000),
        'MLP2': MLPClassifier(hidden_layer_sizes=(200, 50), max_iter=2000),
        'NB': GaussianNB(),
        'QDA': QuadraticDiscriminantAnalysis(),
    }


    feature_sets={
        'mfcc/contrast': columns_dict['mfcc']+columns_dict['spectral_contrast'],
        'mfcc/contrast/chroma': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['chroma_cens'],
        'mfcc/contrast/centroid': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['spectral_centroid'],
        'mfcc/contrast/chroma/centroid': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['chroma_cens']+columns_dict['spectral_centroid'],
        'mfcc/contrast/chroma/centroid/tonnetz': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['chroma_cens']+columns_dict['spectral_centroid']+columns_dict['tonnetz'],
        'mfcc/contrast/chroma/centroid/zcr': columns_dict['mfcc']+columns_dict['spectral_contrast']+columns_dict['chroma_cens']+columns_dict['spectral_centroid']+columns_dict['zcr']
    }
    print("testing...")
    scores, times = test_classifiers_features(classifiers, feature_sets)
    return scores, times

def hyperparams_tuning():
    print("Tuning Parameters...")
    k_dict = {}
    est_dict = {}
    for i in range(50):
        j = 0
        for j in range(199):
            n = j+1
            scores = knn_and_adaboost(n,n,"validate")
            if n in est_dict.keys():
                est_dict[n] += scores["AdaBoost"].mean()
            else:
                est_dict[n] = scores["AdaBoost"].mean()

            if n in k_dict.keys():
                k_dict[n] += scores["kNN"].mean()
            else:
                k_dict[n] = scores["kNN"].mean()
    k_dict = k_dict.update((k, v / 50) for k,v in k_dict.items())
    est_dict = est_dict.update((k, v / 50) for k, v in est_dict.items())
    best_k = max(k_dict, key=k_dict.get)
    best_est = max(est_dict, key=est_dict.get)
    return best_k,best_est

if __name__ == '__main__':
    neighbours, estimators = hyperparams_tuning()  #neighbours for the knn classifier TUNED AND number os estimators for the adaptive boost classifier TUNED
    scores, times = knn_and_adaboost(neighbours,estimators)
    with open("results_KNN_and_ADABOOST_withTunedParams.txt", 'w') as outfile:
        outfile.write("Neighbours: "+str(neighbours)+"\nEstimators: "+str(estimators)+"\n")
        scores.to_string(outfile)
        outfile.write("\n\n\n\n\n\n\n\n")
        times.to_string(outfile)


    #baseline example with all classifiers
    """
    scores, times = baseline_example_all_classifiers()
    with open("results.txt", 'w') as outfile:
        scores.to_string(outfile)
        outfile.write("\n\n\n\n\n\n\n\n")
        times.to_string(outfile)

    """


