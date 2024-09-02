import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle

def load_dataset():
    
    features_path = r'UCI HAR Dataset/features.txt'
    x_train_path = r'UCI HAR Dataset/train/X_train.txt'
    y_train_path = r'UCI HAR Dataset/train/y_train.txt'
    x_test_path = r'UCI HAR Dataset/test/X_test.txt'
    y_test_path = r'UCI HAR Dataset/test/y_test.txt'

    
    features = pd.read_csv(features_path, sep='\s+', header=None, names=['index', 'feature_name'])
    feature_names = features['feature_name'].values

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

    x_train = pd.read_csv(x_train_path, sep='\s+', header=None, names=unique_feature_names)
    y_train = pd.read_csv(y_train_path, sep='\s+', header=None, names=['activity'])
    x_test = pd.read_csv(x_test_path, sep='\s+', header=None, names=unique_feature_names)
    y_test = pd.read_csv(y_test_path, sep='\s+', header=None, names=['activity'])

    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    return train_data, test_data

train_data, test_data = load_dataset()

X_train = train_data.drop('activity', axis=1)
y_train = train_data['activity']
X_test = test_data.drop('activity', axis=1)
y_test = test_data['activity']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_train)

with open('anomaly_detection_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved successfully.")
