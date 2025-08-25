#!/usr/bin/env python3
"""
Install Bi-LSTM fraud detector with TensorFlow
"""

import sys
import subprocess
import platform

def install_tensorflow():
    """Install TensorFlow based on Python version"""
    python_version = sys.version_info
    
    print(f"Python version: {python_version.major}.{python_version.minor}")
    
    if python_version >= (3, 13):
        print("⚠️ Python 3.13+ detected. TensorFlow may not be compatible.")
        print("\nOptions:")
        print("1. Use Python 3.11 or 3.12 (recommended)")
        print("2. Try TensorFlow nightly build (experimental)")
        print("3. Use PyTorch instead")
        
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == "1":
            print("\n📋 To use Python 3.11/3.12:")
            print("1. Install Python 3.11: https://www.python.org/downloads/")
            print("2. Create virtual environment: python3.11 -m venv bilstm_env")
            print("3. Activate: bilstm_env\\Scripts\\activate (Windows)")
            print("4. Install: pip install tensorflow==2.13.0")
            return False
            
        elif choice == "2":
            print("Installing TensorFlow nightly...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-nightly"])
                return True
            except:
                print("❌ Failed to install TensorFlow nightly")
                return False
                
        elif choice == "3":
            print("Installing PyTorch...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
                return True
            except:
                print("❌ Failed to install PyTorch")
                return False
    
    else:
        print("✅ Compatible Python version. Installing TensorFlow...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0"])
            return True
        except:
            print("❌ Failed to install TensorFlow")
            return False

if __name__ == "__main__":
    success = install_tensorflow()
    
    if success:
        print("\n✅ TensorFlow installed! You can now use Bi-LSTM:")
        print("python adaptive_bilstm_fraud.py")
    else:
        print("\n❌ Could not install TensorFlow. Using ensemble models instead.")