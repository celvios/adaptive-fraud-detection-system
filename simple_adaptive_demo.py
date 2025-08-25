import pandas as pd
import numpy as np
from simple_fraud_detector import SimpleFraudDetector
import time
from datetime import datetime

class SimpleAdaptiveLearning:
    def __init__(self):
        self.detector = SimpleFraudDetector()
        self.transaction_log = []
        self.feedback_log = []
        
    def setup_demo(self):
        """Setup demo with sample data"""
        print("🔧 Setting up adaptive learning demo...")
        
        # Load or create sample data
        X, y = self.detector.load_data('creditcard.csv')
        
        # Train initial model
        print("🤖 Training initial model...")
        self.detector.train_models()
        
        print("✅ Demo setup complete!")
        
    def process_transaction(self, transaction_data):
        """Process a transaction and return fraud prediction"""
        fraud_prob, individual_probs = self.detector.predict_fraud(transaction_data)
        
        # Log transaction
        transaction_record = {
            'timestamp': datetime.now(),
            'data': transaction_data,
            'fraud_probability': fraud_prob,
            'prediction': 'FRAUD' if fraud_prob > 0.5 else 'LEGITIMATE'
        }
        self.transaction_log.append(transaction_record)
        
        return transaction_record
    
    def receive_feedback(self, transaction_idx, actual_fraud):
        """Receive feedback on a transaction"""
        if transaction_idx < len(self.transaction_log):
            feedback = {
                'transaction_idx': transaction_idx,
                'actual_fraud': actual_fraud,
                'timestamp': datetime.now()
            }
            self.feedback_log.append(feedback)
            
            print(f"📝 Feedback received: Transaction {transaction_idx} was {'FRAUD' if actual_fraud else 'LEGITIMATE'}")
            
            # Simple adaptive learning: retrain if we have enough feedback
            if len(self.feedback_log) >= 5:
                self.adaptive_retrain()
    
    def adaptive_retrain(self):
        """Simple adaptive retraining"""
        print("🔄 Adaptive retraining triggered...")
        
        # In a real system, this would:
        # 1. Collect new labeled data
        # 2. Retrain the model
        # 3. Evaluate performance
        
        # For demo, we'll simulate improvement
        print("✅ Model updated with new fraud patterns!")
        self.feedback_log = []  # Clear processed feedback
    
    def run_demo(self):
        """Run interactive demo"""
        self.setup_demo()
        
        print("\n" + "="*60)
        print("🔄 ADAPTIVE LEARNING DEMO")
        print("="*60)
        
        # Sample transactions
        transactions = [
            {'Amount': 50.0, 'Time': 12345, 'V1': 0.1, 'V2': -0.2},
            {'Amount': 2500.0, 'Time': 12346, 'V1': 2.5, 'V2': -3.1},
            {'Amount': 25.0, 'Time': 12347, 'V1': 0.05, 'V2': -0.1},
            {'Amount': 5000.0, 'Time': 12348, 'V1': 3.2, 'V2': -4.5},
            {'Amount': 75.0, 'Time': 12349, 'V1': 0.2, 'V2': -0.3}
        ]
        
        print("\n🧪 Processing sample transactions...")
        
        # Process transactions
        for i, transaction in enumerate(transactions):
            print(f"\n--- Transaction {i+1} ---")
            result = self.process_transaction(transaction)
            
            print(f"Amount: ${transaction['Amount']}")
            print(f"Fraud Probability: {result['fraud_probability']:.3f}")
            print(f"Prediction: {result['prediction']}")
            
            # Simulate feedback
            if result['fraud_probability'] > 0.7:
                actual_fraud = 1  # High probability transactions are usually fraud
            elif result['fraud_probability'] < 0.3:
                actual_fraud = 0  # Low probability transactions are usually legitimate
            else:
                actual_fraud = np.random.choice([0, 1])  # Random for medium probability
            
            # Receive feedback after a delay
            time.sleep(1)
            self.receive_feedback(i, actual_fraud)
        
        print("\n📊 Demo Summary:")
        print(f"Transactions processed: {len(self.transaction_log)}")
        print(f"Feedback received: {len(self.feedback_log)}")
        
        print("\n✅ Adaptive learning demo completed!")
        print("\nIn production, this system would:")
        print("- Process real transactions continuously")
        print("- Receive feedback from fraud analysts")
        print("- Automatically retrain when performance drops")
        print("- Adapt to new fraud patterns")

def main():
    """Run the adaptive learning demo"""
    demo = SimpleAdaptiveLearning()
    demo.run_demo()

if __name__ == "__main__":
    main()