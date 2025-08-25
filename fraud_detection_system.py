import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdaptiveFraudDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.explainers = {}
        self.feature_names = None
        self.performance_history = []
        
    def load_data(self, filepath):
        """Load and preprocess the credit card fraud dataset"""
        print("Loading dataset...")
        self.data = pd.read_csv(filepath)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Fraud rate: {self.data['Class'].mean():.4f}")
        
        # Separate features and target
        self.X = self.data.drop('Class', axis=1)
        self.y = self.data['Class']
        self.feature_names = self.X.columns.tolist()
        
        return self.X, self.y
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Split and scale the data"""
        print("Preprocessing data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features (robust to outliers)
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
    def train_ensemble_models(self):
        """Train multiple models for ensemble prediction"""
        print("Training ensemble models...")
        
        # 1. Logistic Regression with SMOTE
        print("Training Logistic Regression...")
        smote_lr = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        smote_lr.fit(self.X_train_scaled, self.y_train)
        self.models['logistic_regression'] = smote_lr
        
        # 2. Random Forest with balanced class weights
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train_scaled, self.y_train)
        self.models['random_forest'] = rf
        
        # 3. XGBoost with scale_pos_weight
        print("Training XGBoost...")
        scale_pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(self.X_train_scaled, self.y_train)
        self.models['xgboost'] = xgb_model
        
        # 4. LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(self.X_train_scaled, self.y_train)
        self.models['lightgbm'] = lgb_model
        
        # 5. Isolation Forest for anomaly detection
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=self.y_train.mean(),
            random_state=42
        )
        iso_forest.fit(self.X_train_scaled[self.y_train == 0])  # Train only on normal transactions
        self.models['isolation_forest'] = iso_forest
        
    def create_explainers(self):
        """Create SHAP explainers for model interpretability"""
        print("Creating SHAP explainers...")
        
        # Create explainers for tree-based models
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            if model_name in self.models:
                self.explainers[model_name] = shap.TreeExplainer(self.models[model_name])
        
        # For logistic regression, use linear explainer
        if 'logistic_regression' in self.models:
            # Get the actual classifier from the pipeline
            lr_classifier = self.models['logistic_regression'].named_steps['classifier']
            self.explainers['logistic_regression'] = shap.LinearExplainer(
                lr_classifier, self.X_train_scaled
            )
    
    def predict_ensemble(self, X, threshold=0.5):
        """Make ensemble predictions"""
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name == 'isolation_forest':
                # Isolation forest returns -1 for anomalies, 1 for normal
                pred = model.predict(X)
                predictions[name] = (pred == -1).astype(int)
                # Convert to probability-like score
                scores = model.decision_function(X)
                probabilities[name] = 1 / (1 + np.exp(scores))  # Sigmoid transformation
            else:
                pred_proba = model.predict_proba(X)[:, 1]
                probabilities[name] = pred_proba
                predictions[name] = (pred_proba > threshold).astype(int)
        
        # Ensemble prediction (majority vote)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        # Ensemble probability (average)
        ensemble_proba = np.mean(list(probabilities.values()), axis=0)
        
        return ensemble_pred_binary, ensemble_proba, predictions, probabilities
    
    def evaluate_models(self):
        """Evaluate all models on test set"""
        print("Evaluating models...")
        
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            if name == 'isolation_forest':
                pred = model.predict(self.X_test_scaled)
                y_pred = (pred == -1).astype(int)
                scores = model.decision_function(self.X_test_scaled)
                y_proba = 1 / (1 + np.exp(scores))
            else:
                y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                y_pred = (y_proba > 0.5).astype(int)
            
            auc_score = roc_auc_score(self.y_test, y_proba)
            results[name] = {
                'auc': auc_score,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"AUC Score: {auc_score:.4f}")
            print(classification_report(self.y_test, y_pred))
        
        # Evaluate ensemble
        ensemble_pred, ensemble_proba, _, _ = self.predict_ensemble(self.X_test_scaled)
        ensemble_auc = roc_auc_score(self.y_test, ensemble_proba)
        
        results['ensemble'] = {
            'auc': ensemble_auc,
            'classification_report': classification_report(self.y_test, ensemble_pred, output_dict=True)
        }
        
        print(f"\nENSEMBLE Results:")
        print(f"AUC Score: {ensemble_auc:.4f}")
        print(classification_report(self.y_test, ensemble_pred))
        
        return results
    
    def explain_prediction(self, transaction_idx, model_name='xgboost'):
        """Explain a specific prediction using SHAP"""
        if model_name not in self.explainers:
            print(f"No explainer available for {model_name}")
            return None
        
        # Get SHAP values for the transaction
        if model_name == 'logistic_regression':
            shap_values = self.explainers[model_name].shap_values(
                self.X_test_scaled[transaction_idx:transaction_idx+1]
            )
        else:
            shap_values = self.explainers[model_name].shap_values(
                self.X_test_scaled[transaction_idx:transaction_idx+1]
            )
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take positive class
        
        return shap_values[0]
    
    def plot_feature_importance(self, model_name='xgboost', top_n=15):
        """Plot feature importance"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif model_name == 'logistic_regression':
            # For pipeline, get the classifier
            classifier = model.named_steps['classifier']
            importances = np.abs(classifier.coef_[0])
        else:
            print(f"Cannot extract feature importance for {model_name}")
            return
        
        # Create feature importance dataframe
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances - {model_name.upper()}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_imp
    
    def adaptive_retrain(self, new_data, retrain_threshold=0.05):
        """Retrain models when performance degrades"""
        print("Checking if retraining is needed...")
        
        # Simulate performance monitoring
        current_performance = self.evaluate_models()
        
        # If this is the first evaluation, store it
        if not self.performance_history:
            self.performance_history.append(current_performance)
            print("Baseline performance stored.")
            return False
        
        # Check if performance has degraded
        baseline_auc = self.performance_history[0]['ensemble']['auc']
        current_auc = current_performance['ensemble']['auc']
        
        performance_drop = baseline_auc - current_auc
        
        if performance_drop > retrain_threshold:
            print(f"Performance dropped by {performance_drop:.4f}. Retraining...")
            
            # Add new data to training set
            if new_data is not None:
                X_new = new_data.drop('Class', axis=1)
                y_new = new_data['Class']
                
                # Combine with existing training data
                self.X_train = pd.concat([self.X_train, X_new])
                self.y_train = pd.concat([self.y_train, y_new])
                
                # Retrain models
                self.preprocess_data()
                self.train_ensemble_models()
                self.create_explainers()
                
                print("Models retrained successfully!")
                return True
        else:
            print(f"Performance is stable. No retraining needed.")
            return False
    
    def save_models(self, filepath_prefix='fraud_model'):
        """Save trained models"""
        for name, model in self.models.items():
            joblib.dump(model, f'{filepath_prefix}_{name}.pkl')
        
        joblib.dump(self.scaler, f'{filepath_prefix}_scaler.pkl')
        print("Models saved successfully!")
    
    def load_models(self, filepath_prefix='fraud_model'):
        """Load trained models"""
        model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'isolation_forest']
        
        for name in model_names:
            try:
                self.models[name] = joblib.load(f'{filepath_prefix}_{name}.pkl')
            except FileNotFoundError:
                print(f"Model {name} not found")
        
        try:
            self.scaler = joblib.load(f'{filepath_prefix}_scaler.pkl')
        except FileNotFoundError:
            print("Scaler not found")
        
        print("Models loaded successfully!")

def main():
    # Initialize the fraud detection system
    detector = AdaptiveFraudDetector()
    
    # Load and preprocess data
    X, y = detector.load_data('creditcard.csv')
    detector.preprocess_data()
    
    # Train ensemble models
    detector.train_ensemble_models()
    
    # Create explainers
    detector.create_explainers()
    
    # Evaluate models
    results = detector.evaluate_models()
    
    # Plot feature importance
    detector.plot_feature_importance('xgboost')
    
    # Example: Explain a fraud prediction
    fraud_indices = detector.y_test[detector.y_test == 1].index[:5]
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        test_idx = list(detector.y_test.index).index(fraud_idx)
        
        print(f"\nExplaining prediction for transaction {fraud_idx}:")
        shap_values = detector.explain_prediction(test_idx, 'xgboost')
        
        if shap_values is not None:
            # Show top contributing features
            feature_contributions = list(zip(detector.feature_names, shap_values))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("Top 10 contributing features:")
            for feature, contribution in feature_contributions[:10]:
                print(f"{feature}: {contribution:.4f}")
    
    # Save models
    detector.save_models()
    
    return detector

if __name__ == "__main__":
    detector = main()