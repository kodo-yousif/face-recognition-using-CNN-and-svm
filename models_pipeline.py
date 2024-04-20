from sklearn.svm import SVC
from get_cnn_model import get_cnn_model
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import Sequential
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier

class CNNFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None
        self.feature_model = None  # We will initialize this only once
    
    def fit(self, X, y=None):
        
        self.model = get_cnn_model()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        self.model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), batch_size=50) 

        self.initialize_feature_model()

        return self

    def transform(self, X):
        feature_model = Sequential(self.model.layers[:-2])
        return feature_model.predict(X)
    
    def initialize_feature_model(self):
        
        if self.model is not None and self.feature_model is None:
            self.feature_model = Sequential(self.model.layers[:-2])

    def transform(self, X):
        if self.feature_model is None:
            self.initialize_feature_model()
        
        return self.feature_model.predict(X)
    


def get_pipeline():
    
    svm_linear = SVC(kernel='linear', probability=True)
    svm_rbf = SVC(kernel='rbf', probability=True)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    gradient_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)

    voting_classifier = VotingClassifier(estimators=[
        ('svm_linear', svm_linear),
        ('svm_rbf', svm_rbf),
        ('random_forest', random_forest),
        ('gradient_boost', gradient_boost)
    ], voting='soft')
    
    return make_pipeline(CNNFeatureExtractor(), OneVsRestClassifier(voting_classifier))
