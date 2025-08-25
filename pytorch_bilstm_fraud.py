import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

class BiLSTMFraudDetector(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(BiLSTMFraudDetector, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc3(out))
        
        return out

class AdaptivePyTorchFraudDetector:
    def __init__(self, sequence_length=20, hidden_size=64):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_sequences(self, X, y=None):
        """Create sequences for LSTM"""
        X_seq = []
        y_seq = [] if y is not None else None
        
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        for i in range(len(X_values) - self.sequence_length + 1):
            X_seq.append(X_values[i:(i + self.sequence_length)])
            if y is not None:
                y_seq.append(y.iloc[i + self.sequence_length - 1])
                
        return np.array(X_seq), np.array(y_seq) if y is not None else None
    
    def train(self, X, y, epochs=100, batch_size=64, learning_rate=0.001):
        """Train the Bi-LSTM model"""
        print("Training PyTorch Bi-LSTM...")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Initialize model
        input_size = X_train.shape[2]
        self.model = BiLSTMFraudDetector(input_size, self.hidden_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    
                    # Calculate AUC
                    val_pred = val_outputs.cpu().numpy()
                    val_true = y_val.cpu().numpy()
                    val_auc = roc_auc_score(val_true, val_pred)
                    
                    print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, '
                          f'Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc:.4f}')
                
                self.model.train()
        
        print("Training completed!")
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
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
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy().flatten()
    
    def save_model(self, filepath='pytorch_bilstm_fraud'):
        """Save model"""
        if self.model is not None:
            torch.save(self.model.state_dict(), f'{filepath}_model.pth')
        joblib.dump(self.scaler, f'{filepath}_scaler.pkl')
        
    def load_model(self, filepath='pytorch_bilstm_fraud', input_size=30):
        """Load model"""
        self.model = BiLSTMFraudDetector(input_size, self.hidden_size).to(self.device)
        self.model.load_state_dict(torch.load(f'{filepath}_model.pth'))
        self.scaler = joblib.load(f'{filepath}_scaler.pkl')

def main():
    """Test PyTorch Bi-LSTM"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    
    # Generate data
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) < 0.1).astype(int)  # 10% fraud
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y)
    
    print(f"Data shape: {X_df.shape}, Fraud rate: {y_series.mean():.3f}")
    
    # Initialize detector
    detector = AdaptivePyTorchFraudDetector(sequence_length=10)
    
    # Train
    detector.train(X_df, y_series, epochs=50)
    
    # Predict
    predictions = detector.predict(X_df)
    
    if len(predictions) > 0:
        # Align labels
        aligned_y = y_series.iloc[detector.sequence_length-1:detector.sequence_length-1+len(predictions)]
        
        auc = roc_auc_score(aligned_y, predictions)
        print(f"\nTest AUC: {auc:.4f}")
        
        # Classification report
        pred_binary = (predictions > 0.5).astype(int)
        print("\nClassification Report:")
        print(classification_report(aligned_y, pred_binary))
    
    # Save model
    detector.save_model()
    
    return detector

if __name__ == "__main__":
    detector = main()