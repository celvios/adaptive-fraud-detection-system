#!/usr/bin/env python3
"""
Quick start script for the Adaptive Credit Card Fraud Detection System
"""

import os
import sys
import subprocess
from fraud_detection_system import AdaptiveFraudDetector

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'imbalanced-learn',
        'xgboost', 'lightgbm', 'shap', 'matplotlib', 'seaborn',
        'plotly', 'streamlit', 'joblib', 'torch', 'schedule'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing packages:", missing_packages)
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Packages installed successfully!")

def run_training_demo():
    """Run a quick training demonstration"""
    print("=" * 60)
    print("ADAPTIVE CREDIT CARD FRAUD DETECTION SYSTEM")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists('creditcard.csv'):
        print("\n❌ Dataset 'creditcard.csv' not found!")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("And place it in the project directory.")
        return False
    
    print("\n🚀 Starting fraud detection system...")
    
    try:
        # Initialize detector
        detector = AdaptiveFraudDetector()
        
        # Load and preprocess data
        print("\n📊 Loading dataset...")
        X, y = detector.load_data('creditcard.csv')
        
        print("\n⚙️ Preprocessing data...")
        detector.preprocess_data()
        
        print("\n🤖 Training ensemble models...")
        detector.train_ensemble_models()
        
        print("\n🔬 Creating explainers...")
        detector.create_explainers()
        
        print("\n📈 Evaluating models...")
        results = detector.evaluate_models()
        
        print("\n💾 Saving models...")
        detector.save_models()
        
        print("\n✅ Training completed successfully!")
        print("\nModel Performance Summary:")
        print("-" * 40)
        
        for model_name, result in results.items():
            auc_score = result['auc']
            
            # Safe access to classification report
            try:
                if '1' in result['classification_report']:
                    precision = result['classification_report']['1']['precision']
                    recall = result['classification_report']['1']['recall']
                else:
                    precision = 0.0
                    recall = 0.0
            except (KeyError, TypeError):
                precision = 0.0
                recall = 0.0
            
            print(f"{model_name.upper():<20} AUC: {auc_score:.4f} | "
                  f"Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("\n🔄 Trying simple fraud detector instead...")
        try:
            from simple_fraud_detector import SimpleFraudDetector
            simple_detector = SimpleFraudDetector()
            X, y = simple_detector.load_data('creditcard.csv')
            simple_detector.train_models()
            results = simple_detector.evaluate_models()
            simple_detector.save_models()
            print("\n✅ Simple fraud detector trained successfully!")
            return True
        except Exception as e2:
            print(f"\n❌ Fallback also failed: {str(e2)}")
            return False

def run_streamlit_app():
    """Launch the Streamlit web application"""
    print("\n🌐 Launching Streamlit web application...")
    print("The app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {str(e)}")

def run_ecommerce_demo():
    """Launch the E-commerce fraud detection demo"""
    print("\n🛒 Launching SecureShop E-commerce Demo...")
    print("Experience real-time fraud detection in action!")
    print("The app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ecommerce_fraud_app.py"])
    except KeyboardInterrupt:
        print("\n👋 E-commerce demo stopped.")
    except Exception as e:
        print(f"\n❌ Error launching e-commerce demo: {str(e)}")

def run_pytorch_bilstm():
    """Train PyTorch Bi-LSTM fraud detector"""
    print("\n🔥 Training PyTorch Bi-LSTM Fraud Detector...")
    print("This will train a bidirectional LSTM neural network for fraud detection.")
    print("Training may take 5-10 minutes depending on your hardware.")
    
    try:
        subprocess.run([sys.executable, "pytorch_bilstm_fraud.py"])
    except Exception as e:
        print(f"\n❌ Error training PyTorch Bi-LSTM: {str(e)}")
        print("Make sure PyTorch is installed: pip install torch")

def run_adaptive_pipeline():
    """Start the adaptive learning pipeline"""
    print("\n🔄 Starting Adaptive Learning Demo...")
    print("This will demonstrate how the system learns from feedback.")
    print("The demo will show:")
    print("- Transaction processing")
    print("- Fraud predictions")
    print("- Feedback collection")
    print("- Adaptive retraining")
    
    try:
        subprocess.run([sys.executable, "simple_adaptive_demo.py"])
    except Exception as e:
        print(f"\n❌ Error starting adaptive demo: {str(e)}")
        print("Running fallback demo...")
        # Fallback to simple demo
        from simple_adaptive_demo import main
        main()

def main():
    """Main function to run the system"""
    print("Adaptive Credit Card Fraud Detection System")
    print("=" * 50)
    
    # Check requirements
    print("Checking requirements...")
    check_requirements()
    
    while True:
        print("\nChoose an option:")
        print("1. Run training demonstration")
        print("2. Launch fraud detection dashboard")
        print("3. Launch SecureShop e-commerce demo")
        print("4. Train PyTorch Bi-LSTM")
        print("5. Start Adaptive Learning Pipeline")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            success = run_training_demo()
            if success:
                print("\n🎉 Training completed! You can now:")
                print("- Launch web app (option 2)")
                print("- Try PyTorch Bi-LSTM (option 4)")
        
        elif choice == '2':
            run_streamlit_app()
        
        elif choice == '3':
            run_ecommerce_demo()
        
        elif choice == '4':
            run_pytorch_bilstm()
        
        elif choice == '5':
            run_adaptive_pipeline()
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")

if __name__ == "__main__":
    main()