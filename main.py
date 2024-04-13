from sklearn.svm import SVC
from load_data import  load_data
from get_cnn_model import get_cnn_model
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

data_dir_path = 'ORLDatabase-2021'

X, y = load_data(data_dir_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


cnn_model = get_cnn_model()

cnn_model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), batch_size=50) 

feature_extractor = Sequential(cnn_model.layers[:-2])


features_train = feature_extractor.predict(X_train)
features_test = feature_extractor.predict(X_test)

svm_model = SVC(kernel='linear')

svm_model.fit(features_train, y_train)

svm_score = svm_model.score(features_test, y_test)

print(f"SVM accuracy: {svm_score}")