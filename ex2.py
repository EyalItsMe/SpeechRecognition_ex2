# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


#euclidian distance with numpy
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
import os
from numba import njit
from hmmlearn import hmm


def euclidian_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

@njit
def dtw_metric(x, y):
    return np.abs(x - y)

#DTW with distance method
@njit
def dtw_distance(x, y, distance_method=dtw_metric):
    n, m = len(x), len(y)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0][0] = distance_method(x[0], y[0])

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = distance_method(x[i-1], y[j-1])
            dtw_matrix[i][j] = cost + min(dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
    return dtw_matrix[n][m]

def load_data(label, train=True):
    directory = f'data/{"train" if train else "test"}/{label}'
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

def extract_features(f_path, feature):
    y, sr = librosa.load(f_path, sr=None)
    if feature == 'raw':
        return y
    elif feature == 'mel':
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        return librosa.power_to_db(mel_spec).flatten()
    elif feature == 'mfcc':
        return librosa.feature.mfcc(y=y, sr=sr).flatten()
    elif feature == 'nmfcc':
        mfcc = librosa.feature.mfcc(y=y, sr=sr).flatten()
        return (mfcc - np.mean(mfcc))/np.std(mfcc)
    return None

def prepare_data(labels, feature, train=True):
    data = [extract_features(f_path, feature) for label in labels for f_path in load_data(label, train)]
    data_label = [label for label in labels for _ in load_data(label, train)]
    return np.array(data), np.array(data_label)

def generate_output_file(feature, labels):
    train_x, train_label = prepare_data(labels, feature, train=True)
    test_x, test_label = prepare_data(labels, feature, train=False)

    knn_euclidean = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_euclidean.fit(train_x, train_label)

    knn_dtw = KNeighborsClassifier(n_neighbors=1, metric=dtw_distance)
    knn_dtw.fit(train_x, train_label)

    euclidean_predictions = knn_euclidean.predict(test_x)
    dtw_predictions = knn_dtw.predict(test_x)

    output_lines = []
    correct_euclidean = 0
    correct_dtw = 0

    pred_index = 0
    for label in labels:
       for i, f_path in enumerate(load_data(label, train=False)):
            file_name = os.path.basename(f_path)
            pred_euclidean = euclidean_predictions[pred_index]
            pred_dtw = dtw_predictions[pred_index]
            output_lines.append(f"{file_name} - {pred_euclidean} - {pred_dtw}")
            print(f"{file_name} - {pred_euclidean} - {pred_dtw}")
            if pred_euclidean == label:
                correct_euclidean += 1
            if pred_dtw == label:
                correct_dtw += 1
            pred_index += 1

    accuracy_euclidean = correct_euclidean / len(test_label)
    accuracy_dtw = correct_dtw / len(test_label)

    output_lines.append(f"Euclidean accuracy: {accuracy_euclidean}")
    output_lines.append(f"DTW accuracy: {accuracy_dtw}")

    print(f"Euclidean accuracy: {accuracy_euclidean}")
    print(f"DTW accuracy: {accuracy_dtw}")
    with open(f'{feature}_output.txt', 'w') as f:
        f.write('\n'.join(output_lines))

def q1(labels):
    features = ['nmfcc', 'mfcc', 'mel', 'raw']

    for feature in features:
         generate_output_file(feature, labels)

def q2(labels):
    '''In this part You should optimize HMM models using hmmlearn package
    you should train a gmm-hmm model for each label independently on top of the mfcc features
    during inference you should classify each test audio file
    according to the model with the highest likelihood to generate this audio file.
    You should explore the different hyper-parameters to reach the best overall performance.
    Similarly to before, generate a file named hmm_output.txt with the predictions for each test file. Each output file should be constructed as
    follows: <filename> - <prediction using gmm-hmm>
    '''
    pass

if __name__ == '__main__':

    labels = ['one', 'two', 'three', 'four', 'five']

    q1(labels)



