import numpy as np
import pandas as pd
from typing import Dict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

class SentimentClassifier:
    def __init__(self, n_neighbors: int = 5):
        self.vectorizer = TfidfVectorizer()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        
    def train(self, X: pd.Series, y: pd.Series) -> Dict:
        """
        Melatih model dengan validasi silang
        """
        # Vektorisasi
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42
        )
        
        # Pelatihan
        self.knn.fit(X_train, y_train)
        
        # Prediksi
        y_pred = self.knn.predict(X_val)
        
        # Evaluasi
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        
        # Cross-validation untuk estimasi performa
        cv_scores = cross_val_score(self.knn, X_vectorized, y, cv=5)
        
        return {
            'model': self.knn,
            'vectorizer': self.vectorizer,
            'accuracy': accuracy,
            'report': report,
            'cv_scores': {
                'mean': np.mean(cv_scores),
                'std': np.std(cv_scores)
            }
        }
    
    def predict(self, texts: pd.Series) -> np.ndarray:
        """
        Memprediksi sentimen untuk teks baru
        """
        X_vectorized = self.vectorizer.transform(texts)
        return self.knn.predict(X_vectorized)