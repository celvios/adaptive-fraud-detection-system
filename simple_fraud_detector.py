import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

class SimpleFraudDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load credit card fraud dataset"""
        try:
            self.data = pd.read_csv(filepath)
            self.X = self.data.drop('Class', axis=1)
            self.y = self.data['Class']
            self.feature_names = self.X.columns.tolist()
            return self.X, self.y
        except FileNotFoundError:
            # Create sample data if file not found
            print("Creating sample fraud detection data...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample fraud data for demo"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 30
        
        # Generate normal transactions
        normal_data = np.random.normal(0, 1, (int(n_samples * 0.99), n_features))
        normal_labels = np.zeros(int(n_samples * 0.99))
        
        # Generate fraud transactions (more extreme values)
        fraud_data = np.random.normal(0, 3, (int(n_samples * 0.01), n_features))
        fraud_labels = np.ones(int(n_samples * 0.01))
        
        # Combine data
        X = np.vstack([normal_data, fraud_data])
        y = np.hstack([normal_labels, fraud_labels])
        
        # Create feature names
        feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        
        self.X = pd.DataFrame(X, columns=feature_names)
        self.y = pd.Series(y)
        self.feature_names = feature_names
        
        print(f"Sample data created: {len(self.X)} transactions, {self.y.sum()} frauds")
        return self.X, self.y
    
    def train_models(self):
        """Train fraud detection models"""
        print("Training fraud detection models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # 1. Logistic Regression with SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
        
        lr = LogisticRegression(random_state=42)
        lr.fit(X_smote, y_smote)
        self.models['logistic_regression'] = lr
        
        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf
        
        # 3. Isolation Forest
        iso = IsolationForest(contamination=y_train.mean(), random_state=42)
        iso.fit(X_train_scaled[y_train == 0])  # Train on normal transactions only
        self.models['isolation_forest'] = iso
        
        print("Models trained successfully!")
        
    def predict_fraud(self, transaction_data):
        """Predict fraud probability for transaction"""
        if isinstance(transaction_data, dict):
            # Convert dict to DataFrame
            transaction_df = pd.DataFrame([transaction_data])
        else:
            transaction_df = transaction_data
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in transaction_df.columns:
                transaction_df[feature] = 0.0
        
        # Scale features
        transaction_scaled = self.scaler.transform(transaction_df[self.feature_names])
        
        # Get predictions from all models
        predictions = {}
        
        # Logistic Regression
        lr_prob = self.models['logistic_regression'].predict_proba(transaction_scaled)[:, 1]
        predictions['logistic_regression'] = lr_prob[0]
        
        # Random Forest
        rf_prob = self.models['random_forest'].predict_proba(transaction_scaled)[:, 1]
        predictions['random_forest'] = rf_prob[0]
        
        # Isolation Forest
        iso_score = self.models['isolation_forest'].decision_function(transaction_scaled)
        iso_prob = 1 / (1 + np.exp(iso_score[0]))  # Convert to probability
        predictions['isolation_forest'] = iso_prob
        
        # Ensemble prediction (average)
        ensemble_prob = np.mean(list(predictions.values()))
        
        return ensemble_prob, predictions
    
    def evaluate_models(self):
        """Evaluate model performance"""
        print("Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            if name == 'isolation_forest':
                pred = model.predict(self.X_test)
                y_pred = (pred == -1).astype(int)
                scores = model.decision_function(self.X_test)
                y_proba = 1 / (1 + np.exp(scores))
            else:
                y_proba = model.predict_proba(self.X_test)[:, 1]
                y_pred = (y_proba > 0.5).astype(int)
            
            auc_score = roc_auc_score(self.y_test, y_proba)
            results[name] = {
                'auc': auc_score,
                'report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            print(f"\n{name.upper()}:")
            print(f"AUC: {auc_score:.4f}")
            print(classification_report(self.y_test, y_pred))
        
        return results
    
    def save_models(self, prefix='simple_fraud'):
        """Save trained models"""
        for name, model in self.models.items():
            joblib.dump(model, f'{prefix}_{name}.pkl')
        joblib.dump(self.scaler, f'{prefix}_scaler.pkl')
        print("Models saved successfully!")
    
    def load_models(self, prefix='simple_fraud'):
        """Load trained models"""
        try:
            self.models['logistic_regression'] = joblib.load(f'{prefix}_logistic_regression.pkl')
            self.models['random_forest'] = joblib.load(f'{prefix}_random_forest.pkl')
            self.models['isolation_forest'] = joblib.load(f'{prefix}_isolation_forest.pkl')
            self.scaler = joblib.load(f'{prefix}_scaler.pkl')
            print("Models loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")

def main():
    """Test the fraud detector"""
    detector = SimpleFraudDetector()
    
    # Load data
    X, y = detector.load_data('creditcard.csv')
    
    # Train models
    detector.train_models()
    
    # Evaluate
    results = detector.evaluate_models()
    
    # Test prediction
    sample_transaction = {
        'Time': 12345,
        'Amount': 150.0,
        'V1': 0.5,
        'V2': -0.3
    }
    
    fraud_prob, individual_probs = detector.predict_fraud(sample_transaction)
    
    print(f"\nSample Transaction Analysis:")
    print(f"Fraud Probability: {fraud_prob:.3f}")
    print(f"Individual Model Predictions: {individual_probs}")
    
    if fraud_prob > 0.7:
        print("🚨 HIGH RISK - Transaction flagged for review")
    elif fraud_prob > 0.3:
        print("⚠️ MEDIUM RISK - Additional monitoring recommended")
    else:
        print("✅ LOW RISK - Transaction approved")
    
    # Save models
    detector.save_models()
    
    return detector

if __name__ == "__main__":
    detector = main()