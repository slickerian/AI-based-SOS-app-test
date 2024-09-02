import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle

def load_model(model_path='anomaly_detection_model.pkl'):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def preprocess_data(data, scaler):
    
    return scaler.transform(data)

def detect_anomalies(data, model, scaler):
    processed_data = preprocess_data(data, scaler)
    predictions = model.predict(processed_data)
    return predictions

def make_feature_names_unique(feature_names):
    seen = set()
    unique_feature_names = []
    for name in feature_names:
        new_name = name
        i = 1
        while new_name in seen:
            new_name = f"{name}_duplicate_{i}"
            i += 1
        seen.add(new_name)
        unique_feature_names.append(new_name)
    return unique_feature_names

def main():
    model_path = 'anomaly_detection_model.pkl'
    features_path = r'UCI HAR Dataset/features.txt'
    x_test_path = r'UCI HAR Dataset/test/X_test.txt'
    y_test_path = r'UCI HAR Dataset/test/y_test.txt'

    features = pd.read_csv(features_path, sep='\s+', header=None, names=['index', 'feature_name'])
    feature_names = features['feature_name'].values
    unique_feature_names = make_feature_names_unique(feature_names)

    x_test = pd.read_csv(x_test_path, sep='\s+', header=None, names=unique_feature_names)
    y_test = pd.read_csv(y_test_path, sep='\s+', header=None, names=['activity'])

    model = load_model(model_path)

    scaler = StandardScaler()
    scaler.fit(x_test)  

    
    predictions = detect_anomalies(x_test, model, scaler)

    anomalies = x_test[predictions == -1]
    normal = x_test[predictions == 1]

    print(f"Number of anomalies detected: {len(anomalies)}")
    print(f"Number of normal movements detected: {len(normal)}")

    anomalies.to_csv('anomalies_detected.csv', index=False)
    print("Anomalies have been saved to 'anomalies_detected.csv'")

if __name__ == "__main__":
    main()
