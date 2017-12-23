import time

import numpy as np
import pandas as pd
from time import time as tm, gmtime, strftime
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix



def load_training_parameters():
    print("Loading the files...")
    # pd.set_option('max_columns',20)
    tracks = pd.read_csv('tracks.csv', low_memory=False, skiprows=[1])
    features = pd.read_csv('features.csv', low_memory=False, skiprows=[1, 2])

    columns_dict = {}
    for column in features.columns:
        if '.' in column and column.split('.')[0] in columns_dict.keys():
            columns_dict[column.split('.')[0]].append(column)
        else:
            columns_dict[column] = [column]

    subset = tracks.index[(tracks["set.1"] == 'medium') | (tracks["set.1"] == 'small')]

    tracks = tracks.loc[subset]
    features_all = features.loc[subset]

    train = tracks.index[tracks['set'] == 'training']
    val = tracks.index[tracks['set'] == 'validation']
    test = tracks.index[tracks['set'] == 'test']

    print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))


    print('Top genres ({}): {}'.format(len(list(LabelEncoder().fit(tracks['track.7']).classes_)), list(LabelEncoder().fit(tracks['track.7']).classes_)))

    feature_sets = {
        'mfcc/contrast': columns_dict['mfcc'] + columns_dict['spectral_contrast'],
        'mfcc/contrast/chroma': columns_dict['mfcc'] + columns_dict['spectral_contrast'] + columns_dict['chroma_cens'],
        'mfcc/contrast/centroid': columns_dict['mfcc'] + columns_dict['spectral_contrast'] +
                                  columns_dict['spectral_centroid'],
        'mfcc/contrast/chroma/centroid': columns_dict['mfcc'] + columns_dict['spectral_contrast'] +
                                         columns_dict['chroma_cens'] + columns_dict['spectral_centroid'],
        'mfcc/contrast/chroma/centroid/tonnetz': columns_dict['mfcc'] + columns_dict['spectral_contrast'] +
                                                 columns_dict['chroma_cens'] + columns_dict['spectral_centroid'] +
                                                 columns_dict['tonnetz'],
        'mfcc/contrast/chroma/centroid/zcr': columns_dict['mfcc'] + columns_dict['spectral_contrast'] +
                                             columns_dict['chroma_cens'] + columns_dict['spectral_centroid'] +
                                             columns_dict['zcr']
    }

    return tracks, features_all, feature_sets


def knn_and_adaboost(tracks, features_all, feature_sets, neighbours, estimators, op="test"):

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
        # print("Spliting the dataset in training, validation and testing sets...")
        y_train = enc.fit_transform(labels[train])
        # print(y_train)
        y_val = enc.transform(labels[val])
        y_test = enc.transform(labels[test])
        x_train = (features.loc[train, columns]).as_matrix()
        x_val = features.loc[val, columns].as_matrix()
        x_test = features.loc[test, columns].as_matrix()

        x_train, y_train = shuffle(x_train, y_train, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance.
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(x_train)
        scaler.transform(x_val)
        scaler.transform(x_test)

        return y_train, y_val, y_test, x_train, x_val, x_test

    def validate_classifiers_features(classifiers, feature_sets, multi_label=False):
        columns = list(classifiers.keys()).insert(0, 'dim')
        scores = pd.DataFrame(columns=columns, index=feature_sets.keys())
        for fset_name, fset in feature_sets.items():
            y_train, y_val, y_test, x_train, x_val, x_test = pre_process(tracks, features_all, fset, multi_label)
            scores.loc[fset_name, 'dim'] = x_train.shape[1]
            for clf_name, clf in classifiers.items():
                clf.fit(x_train, y_train)
                score = clf.score(x_val, y_val)
                scores.loc[fset_name, clf_name] = score
        return scores

    def test_classifiers_features(classifiers, feature_sets, multi_label=False):
        matrixes = {} #confusion matrixes
        #genres = list(LabelEncoder().fit(tracks['track.7']).classes_)
        columns = list(classifiers.keys()).insert(0, 'dim')
        scores = pd.DataFrame(columns=columns, index=feature_sets.keys())
        times = pd.DataFrame(columns=classifiers.keys(), index=feature_sets.keys())
        start = tm()
        for fset_name, fset in feature_sets.items():
            start_comb = tm()
            y_train, y_val, y_test, x_train, x_val, x_test = pre_process(tracks, features_all, fset, multi_label)
            print("Combination: {}".format(fset_name))
            scores.loc[fset_name, 'dim'] = x_train.shape[1]
            for clf_name, clf in classifiers.items():
                start_classifier = tm()
                print("\tClassifier: {}".format(clf_name))
                t = time.process_time()
                clf.fit(x_train, y_train)
                score = clf.score(x_test, y_test)
                scores.loc[fset_name, clf_name] = score
                times.loc[fset_name, clf_name] = time.process_time() - t
                print("\tTime: {} s".format(tm() - start_classifier))
                conf_matrix = confusion_matrix(y_test,y_train)
                matrixes[clf_name] = conf_matrix
            print("Time for {}: {} s".format(fset_name, tm() - start_comb))
        print("Test Classifiers Features Finish.")
        print("Total time: {}".format(tm() - start))
        return scores, times, matrixes

    train = tracks.index[tracks['set'] == 'training']
    val = tracks.index[tracks['set'] == 'validation']
    test = tracks.index[tracks['set'] == 'test']

    #print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

    #genres = list(LabelEncoder().fit(tracks['track.7']).classes_)

    #print('Top genres ({}): {}'.format(len(genres), genres))

    classifiers = {
        'kNN': KNeighborsClassifier(n_neighbors=neighbours),
        'AdaBoost': AdaBoostClassifier(n_estimators=estimators),
    }

    if op == 'validate':
        scores = validate_classifiers_features(classifiers, feature_sets)
        return scores

    scores, times = test_classifiers_features(classifiers, feature_sets)
    return scores, times


def hyperparams_tuning(tracks, features_all, feature_sets, trials):
    print("Tuning Parameters...")
    k_dict = {}
    est_dict = {}
    for i in range(trials):
        print("Trial {} start time: {}".format(i + 1, strftime("%H:%M:%S %d-%m-%Y", gmtime())))
        it = 0
        for j in range(0, 110, 10):
            it += 1
            print("\r{}/11".format(it), end="")
            k = j
            est = int(k/10)
            if j == 0:
                k = 1
                est = 1
            scores = knn_and_adaboost(tracks, features_all, feature_sets, k, est, "validate")
            if est in est_dict.keys():
                est_dict[est] += scores["AdaBoost"].mean()
            else:
                est_dict[est] = scores["AdaBoost"].mean()

            if k in k_dict.keys():
                k_dict[k] += scores["kNN"].mean()
            else:
                k_dict[k] = scores["kNN"].mean()
        print()

    test_k_dict = {key: value / trials for key, value in k_dict.items()}
    test_est_dict = {key: value / trials for key, value in est_dict.items()}
    best_k = max(test_k_dict, key=test_k_dict.get)
    best_est = max(test_est_dict, key=test_est_dict.get)
    return best_k, best_est


def main():
    tracks, features_all, feature_sets = load_training_parameters()
    # neighbours for the knn classifier TUNED AND number os estimators for the adaptive boost classifier TUNED
    trials = 25
    fine_neighbours, fine_estimators = hyperparams_tuning(tracks, features_all, feature_sets, trials)
    print("Fine Nieghbours:", fine_neighbours, "Fine Estimators:", fine_estimators)
    scores, times, confusion_matrixes = knn_and_adaboost(tracks, features_all, feature_sets, fine_neighbours, fine_estimators)
    with open("results_kNN_and_ADABOOST_withTunedParams.txt", 'w') as outfile:
        outfile.write("Neighbours: " + str(fine_neighbours) + "\nEstimators: " + str(fine_estimators) + "\n")
        scores.to_string(outfile)
        outfile.write("\n\n\n\n\n\n\n\n")
        times.to_string(outfile)
        outfile.write("\n\n\n\n\n\n\n\n")
        outfile.write("Confusion Matrixes\nkNN:\n"+str(confusion_matrixes["kNN"])+"\nAdaBoost:\n"+str(confusion_matrixes["AdaBoost"]))
        outfile.write()


if __name__ == '__main__':
    main()
