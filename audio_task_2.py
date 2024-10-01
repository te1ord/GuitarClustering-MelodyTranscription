import sys
import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_combined_features(audio_folder_path):

    features = []
    file_names = []

    for file in os.listdir(audio_folder_path):

        if file.endswith('.wav'):

            file_path = os.path.join(audio_folder_path, file)
            y, sr = librosa.load(file_path, sr=None)

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
           
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr, axis=1)

    
            combined_features = np.concatenate([
                mfccs_mean, spectral_contrast_mean,zcr_mean 
                ])
            
            features.append(combined_features)
            file_names.append(file)

    return np.array(features), file_names

def cluster_audio_files(features, file_names):

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(scaled_features)
    labels = kmeans.labels_

    clusters = {i: [] for i in range(2)}

    for file, label in zip(file_names, labels):
        clusters[label].append(file)

    output = [(tuple(cluster)) for cluster in clusters.values()]

    return output

def main(audio_folder_path):
    features, file_names = extract_combined_features(audio_folder_path)
    clusters = cluster_audio_files(features, file_names)
    print(clusters)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python audio_task_2.py <path_to_audio_folder>")
        sys.exit(1)
    audio_folder_path = sys.argv[1]
    main(audio_folder_path)
