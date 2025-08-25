import pandas as pd
import numpy as np
import torch
import sqlite3
import schedule
import time
from datetime import datetime, timedelta
from pytorch_bilstm_fraud import AdaptivePyTorchFraudDetector
import threading
import queue
import json

class AdaptiveLearningPipeline:
    def __init__(self, model_path='pytorch_bilstm_fraud'):
        self.model = AdaptivePyTorchFraudDetector()
        self.model_path = model_path
        self.learning_queue = queue.Queue()
        self.performance_threshold = 0.05
        self.min_batch_size = 50
        
        # Initialize database for storing transactions
        self.init_database()
        
        # Load existing model if available
        try:
            self.model.load_model(model_path)
            print("✅ Loaded existing model")
        except:
            print("⚠️ No existing model found. Will train new model.")
    
    def init_database(self):
        """Initialize SQLite database for transaction storage"""
        self.conn = sqlite3.connect('fraud_transactions.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                transaction_data TEXT,
                predicted_fraud REAL,
                actual_fraud INTEGER,
                confidence REAL,
                feedback_received BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                auc_score REAL,
                precision_score REAL,
                recall_score REAL,
                model_version TEXT
            )
        ''')
        
        self.conn.commit()
        print("✅ Database initialized")
    
    def process_transaction(self, transaction_data, user_id=None):
        """Process a single transaction with fraud detection"""
        try:
            # Convert transaction to DataFrame
            if isinstance(transaction_data, dict):
                df = pd.DataFrame([transaction_data])
            else:
                df = transaction_data
            
            # Make prediction
            fraud_probability = self.model.predict(df)
            
            if len(fraud_probability) == 0:
                fraud_probability = [0.5]  # Default if no prediction
            
            fraud_prob = float(fraud_probability[0])
            confidence = abs(fraud_prob - 0.5) * 2  # 0-1 confidence score
            
            # Store transaction in database
            transaction_json = json.dumps(transaction_data)
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO transactions 
                (transaction_data, predicted_fraud, confidence)
                VALUES (?, ?, ?)
            ''', (transaction_json, fraud_prob, confidence))
            
            transaction_id = cursor.lastrowid
            self.conn.commit()
            
            # Determine risk level
            if fraud_prob > 0.7:
                risk_level = "HIGH"
                action = "BLOCK_AND_VERIFY"
            elif fraud_prob > 0.4:
                risk_level = "MEDIUM"
                action = "ADDITIONAL_MONITORING"
            else:
                risk_level = "LOW"
                action = "APPROVE"
            
            result = {
                'transaction_id': transaction_id,
                'fraud_probability': fraud_prob,
                'risk_level': risk_level,
                'confidence': confidence,
                'recommended_action': action,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"🔍 Transaction {transaction_id}: {risk_level} risk ({fraud_prob:.3f})")
            return result
            
        except Exception as e:
            print(f"❌ Error processing transaction: {e}")
            return {'error': str(e)}
    
    def receive_feedback(self, transaction_id, actual_fraud_label, feedback_source="analyst"):
        """Receive feedback on transaction and queue for learning"""
        try:
            # Update database with actual label
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE transactions 
                SET actual_fraud = ?, feedback_received = TRUE
                WHERE id = ?
            ''', (int(actual_fraud_label), transaction_id))
            self.conn.commit()
            
            # Get transaction data for learning
            cursor.execute('''
                SELECT transaction_data, predicted_fraud, actual_fraud
                FROM transactions WHERE id = ?
            ''', (transaction_id,))
            
            result = cursor.fetchone()
            if result:
                transaction_data = json.loads(result[0])
                predicted = result[1]
                actual = result[2]
                
                # Add to learning queue
                self.learning_queue.put({
                    'transaction_data': transaction_data,
                    'actual_label': actual,
                    'predicted': predicted,
                    'source': feedback_source,
                    'timestamp': datetime.now()
                })
                
                print(f"📝 Feedback received for transaction {transaction_id}: {'FRAUD' if actual else 'LEGITIMATE'}")
                
                # Trigger immediate learning for high-confidence errors
                prediction_error = abs(predicted - actual)
                if prediction_error > 0.5:  # Major error
                    self.immediate_learning()
                
                return True
            
        except Exception as e:
            print(f"❌ Error receiving feedback: {e}")
            return False
    
    def immediate_learning(self):
        """Immediate learning from recent high-confidence feedback"""
        try:
            if self.learning_queue.qsize() > 0:
                print("🔄 Immediate learning triggered...")
                
                # Get recent feedback
                recent_feedback = []
                while not self.learning_queue.empty() and len(recent_feedback) < 10:
                    recent_feedback.append(self.learning_queue.get())
                
                if len(recent_feedback) >= 3:  # Minimum for immediate learning
                    self.train_on_feedback(recent_feedback)
                    print("✅ Immediate learning completed")
                else:
                    # Put back in queue if not enough data
                    for feedback in recent_feedback:
                        self.learning_queue.put(feedback)
                        
        except Exception as e:
            print(f"❌ Error in immediate learning: {e}")
    
    def batch_learning(self):
        """Batch learning from accumulated feedback"""
        try:
            print("🔄 Starting batch learning...")
            
            # Get all pending feedback
            feedback_batch = []
            while not self.learning_queue.empty():
                feedback_batch.append(self.learning_queue.get())
            
            # Get additional feedback from database
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT transaction_data, actual_fraud
                FROM transactions 
                WHERE feedback_received = TRUE 
                AND timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
                LIMIT 1000
            ''')
            
            db_feedback = cursor.fetchall()
            for row in db_feedback:
                feedback_batch.append({
                    'transaction_data': json.loads(row[0]),
                    'actual_label': row[1],
                    'source': 'database'
                })
            
            if len(feedback_batch) >= self.min_batch_size:
                self.train_on_feedback(feedback_batch)
                print(f"✅ Batch learning completed with {len(feedback_batch)} samples")
            else:
                print(f"⏳ Not enough feedback for batch learning ({len(feedback_batch)}/{self.min_batch_size})")
                
        except Exception as e:
            print(f"❌ Error in batch learning: {e}")
    
    def train_on_feedback(self, feedback_batch):
        """Train model on feedback data"""
        try:
            # Prepare training data
            X_data = []
            y_data = []
            
            for feedback in feedback_batch:
                transaction = feedback['transaction_data']
                label = feedback['actual_label']
                
                # Convert to feature vector
                feature_vector = self.transaction_to_features(transaction)
                X_data.append(feature_vector)
                y_data.append(label)
            
            if len(X_data) > 0:
                X_df = pd.DataFrame(X_data)
                y_series = pd.Series(y_data)
                
                # Fine-tune model with lower learning rate
                print(f"🧠 Training on {len(X_data)} feedback samples...")
                self.model.train(X_df, y_series, epochs=10, learning_rate=0.0001)
                
                # Save updated model
                self.model.save_model(self.model_path)
                
                # Log performance
                self.log_performance()
                
        except Exception as e:
            print(f"❌ Error training on feedback: {e}")
    
    def transaction_to_features(self, transaction):
        """Convert transaction dict to feature vector"""
        # Create feature vector with default values
        features = {}
        
        # Standard features
        features['Time'] = transaction.get('Time', 0)
        features['Amount'] = transaction.get('Amount', 0)
        
        # V1-V28 features (PCA components)
        for i in range(1, 29):
            features[f'V{i}'] = transaction.get(f'V{i}', 0)
        
        return features
    
    def log_performance(self):
        """Log current model performance"""
        try:
            # Get recent transactions with feedback
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT predicted_fraud, actual_fraud
                FROM transactions 
                WHERE feedback_received = TRUE 
                AND timestamp > datetime('now', '-1 days')
            ''')
            
            results = cursor.fetchall()
            if len(results) > 10:
                predictions = [r[0] for r in results]
                actuals = [r[1] for r in results]
                
                # Calculate metrics
                from sklearn.metrics import roc_auc_score, precision_score, recall_score
                
                auc = roc_auc_score(actuals, predictions)
                precision = precision_score(actuals, [1 if p > 0.5 else 0 for p in predictions])
                recall = recall_score(actuals, [1 if p > 0.5 else 0 for p in predictions])
                
                # Store performance
                cursor.execute('''
                    INSERT INTO model_performance (auc_score, precision_score, recall_score, model_version)
                    VALUES (?, ?, ?, ?)
                ''', (auc, precision, recall, f"v{datetime.now().strftime('%Y%m%d_%H%M')}"))
                self.conn.commit()
                
                print(f"📊 Performance: AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
                
        except Exception as e:
            print(f"❌ Error logging performance: {e}")
    
    def start_adaptive_pipeline(self):
        """Start the adaptive learning pipeline"""
        print("🚀 Starting Adaptive Learning Pipeline...")
        
        # Schedule batch learning
        schedule.every().day.at("02:00").do(self.batch_learning)
        schedule.every().hour.do(self.immediate_learning)
        
        # Start scheduler in background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        print("✅ Adaptive learning pipeline started!")
        print("- Immediate learning: Every hour")
        print("- Batch learning: Daily at 2:00 AM")
        print("- Feedback processing: Real-time")

def main():
    """Demo the adaptive learning pipeline"""
    pipeline = AdaptiveLearningPipeline()
    
    # Start the pipeline
    pipeline.start_adaptive_pipeline()
    
    # Simulate some transactions
    print("\n🧪 Simulating transactions...")
    
    # Normal transaction
    normal_transaction = {
        'Time': 12345,
        'Amount': 50.0,
        'V1': 0.1, 'V2': -0.2, 'V3': 0.3,
        'V4': 0.1, 'V5': -0.1, 'V6': 0.2
    }
    
    result1 = pipeline.process_transaction(normal_transaction)
    print(f"Result 1: {result1}")
    
    # Suspicious transaction
    suspicious_transaction = {
        'Time': 12346,
        'Amount': 2500.0,
        'V1': 2.5, 'V2': -3.1, 'V3': 1.8,
        'V4': 2.2, 'V5': -2.8, 'V6': 3.5
    }
    
    result2 = pipeline.process_transaction(suspicious_transaction)
    print(f"Result 2: {result2}")
    
    # Simulate feedback
    print("\n📝 Simulating feedback...")
    pipeline.receive_feedback(result1['transaction_id'], 0, "customer_service")  # Not fraud
    pipeline.receive_feedback(result2['transaction_id'], 1, "fraud_analyst")     # Confirmed fraud
    
    print("\n✅ Demo completed! Pipeline is running in background.")
    print("In production, this would:")
    print("- Process real transactions")
    print("- Receive feedback from analysts")
    print("- Learn continuously")
    print("- Improve over time")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()
    
    # Keep running for demo
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n👋 Pipeline stopped.")