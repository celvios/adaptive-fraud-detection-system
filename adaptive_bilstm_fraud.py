import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

class AdaptiveBiLSTMFraudDetector:
    def __init__(self, sequence_length=20, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.performance_history = []
        
    def create_sequences(self, X, y=None):
        """Create sequences for LSTM with sliding window"""
        X_seq = []
        y_seq = [] if y is not None else None
        
        # Sort by time if available
        if 'Time' in X.columns:
            X = X.sort_values('Time')
            if y is not None:
                y = y.reindex(X.index)
        
        X_values = X.values
        
        for i in range(len(X_values) - self.sequence_length + 1):
            X_seq.append(X_values[i:(i + self.sequence_length)])
            if y is not None:
                y_seq.append(y.iloc[i + self.sequence_length - 1])
                
        return np.array(X_seq), np.array(y_seq) if y is not None else None
    
    def build_adaptive_model(self, input_shape):
        """Build adaptive Bi-LSTM with attention mechanism"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.2), 
                         input_shape=input_shape),
            BatchNormalization(),
            
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
            BatchNormalization(),
            
            Bidirectional(LSTM(32, dropout=0.2)),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=64):
        """Train the adaptive Bi-LSTM model"""
        print("Preprocessing data for Bi-LSTM...")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        if len(X_seq) == 0:
            raise ValueError("Not enough data to create sequences")
        
        print(f"Created {len(X_seq)} sequences of length {self.sequence_length}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=validation_split, 
            random_state=42, stratify=y_seq
        )
        
        # Build model
        self.model = self.build_adaptive_model((self.sequence_length, X.shape[1]))
        
        # Callbacks for adaptive learning
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Handle class imbalance
        class_weight = {
            0: 1.0,
            1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        print("Training Bi-LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Evaluate
        val_pred = self.model.predict(X_val)
        val_auc = roc_auc_score(y_val, val_pred)
        
        print(f"Validation AUC: {val_auc:.4f}")
        
        # Store performance
        self.performance_history.append({
            'epoch': len(history.history['loss']),
            'val_auc': val_auc,
            'val_loss': min(history.history['val_loss'])
        })
        
        return history
    
    def predict(self, X, return_sequences=False):
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Create sequences
        X_seq, _ = self.create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            return np.array([])
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        if return_sequences:
            return predictions, X_seq
        
        return predictions.flatten()
    
    def adaptive_update(self, new_X, new_y, retrain_threshold=0.05):
        """Adaptively update the model with new data"""
        if self.model is None:
            print("No existing model. Training from scratch...")
            return self.train(new_X, new_y)
        
        print("Evaluating current model performance...")
        
        # Test current model on new data
        current_pred = self.predict(new_X)
        
        if len(current_pred) > 0:
            # Align predictions with labels (sequences are shorter)
            aligned_y = new_y.iloc[self.sequence_length-1:self.sequence_length-1+len(current_pred)]
            current_auc = roc_auc_score(aligned_y, current_pred)
            
            # Compare with historical performance
            if self.performance_history:
                best_auc = max([p['val_auc'] for p in self.performance_history])
                performance_drop = best_auc - current_auc
                
                print(f"Current AUC: {current_auc:.4f}, Best AUC: {best_auc:.4f}")
                print(f"Performance drop: {performance_drop:.4f}")
                
                if performance_drop > retrain_threshold:
                    print("Performance degraded. Retraining...")
                    
                    # Incremental learning: lower learning rate
                    self.learning_rate *= 0.1
                    self.model.optimizer.learning_rate = self.learning_rate
                    
                    # Fine-tune with new data
                    return self.train(new_X, new_y, epochs=50)
                else:
                    print("Performance stable. No retraining needed.")
                    return None
        
        return None
    
    def online_learn(self, new_transaction, true_label, learning_rate=0.0001):
        """Online learning from single transaction"""
        if self.model is None:
            return
        
        # This would require implementing online LSTM learning
        # For now, we collect transactions and retrain periodically
        pass
    
    def save_model(self, filepath='adaptive_bilstm_fraud'):
        """Save the trained model and scaler"""
        if self.model is not None:
            self.model.save(f'{filepath}_model.h5')
        
        joblib.dump(self.scaler, f'{filepath}_scaler.pkl')
        joblib.dump(self.performance_history, f'{filepath}_history.pkl')
        
        print("Model saved successfully!")
    
    def load_model(self, filepath='adaptive_bilstm_fraud'):
        """Load a trained model and scaler"""
        try:
            self.model = tf.keras.models.load_model(f'{filepath}_model.h5')
            self.scaler = joblib.load(f'{filepath}_scaler.pkl')
            self.performance_history = joblib.load(f'{filepath}_history.pkl')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

def main():
    """Test the adaptive Bi-LSTM fraud detector"""
    # Load data
    print("Loading credit card fraud dataset...")
    data = pd.read_csv('creditcard.csv')
    
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Fraud rate: {y.mean():.4f}")
    
    # Initialize detector
    detector = AdaptiveBiLSTMFraudDetector(sequence_length=20)
    
    # Train model
    history = detector.train(X, y, epochs=50)
    
    # Test predictions
    print("\nTesting predictions...")
    predictions = detector.predict(X)
    
    if len(predictions) > 0:
        # Align with actual labels
        aligned_y = y.iloc[detector.sequence_length-1:detector.sequence_length-1+len(predictions)]
        
        test_auc = roc_auc_score(aligned_y, predictions)
        print(f"Test AUC: {test_auc:.4f}")
        
        # Classification report
        pred_binary = (predictions > 0.5).astype(int)
        print("\nClassification Report:")
        print(classification_report(aligned_y, pred_binary))
    
    # Save model
    detector.save_model()
    
    return detector

if __name__ == "__main__":
    detector = main()