"""
IDS with Machine Learning - CLI Tool
=====================================
Loads trained ML models (Random Forest / XGBoost) for intrusion detection.
Uses: scaler.pkl, label_encoder.pkl, config.json, and model files.
"""

import os
import json
import pickle
import warnings

# Suppress sklearn version warnings (models work fine across versions)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*unpickle.*')

import numpy as np
import pandas as pd

# ============================================================
# IDSModel Class - Handles model loading and prediction
# ============================================================

class IDSModel:
    """
    IDS Model wrapper for loading and using trained ML models.
    """
    
    def __init__(self, models_dir=None):
        """
        Initialize IDS Model.
        
        Args:
            models_dir: Path to models directory (default: ./models)
        """
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        self.models_dir = models_dir
        self.model = None
        self.model_name = None
        self.scaler = None
        self.label_encoder = None
        self.config = None
        self.features = None
        self.classes = None
        self.loaded = False
    
    def load(self, model_name=None):
        """
        Load model and associated files.
        
        Args:
            model_name: 'random_forest' or 'xgboost' (default: from config.json best_model)
        
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Load config
            config_path = os.path.join(self.models_dir, 'config.json')
            if not os.path.exists(config_path):
                print(f"❌ Config file not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            self.features = self.config.get('features', [])
            self.classes = self.config.get('classes', [])
            
            # Determine which model to load
            if model_name is None:
                model_name = self.config.get('best_model', 'random_forest')
            
            self.model_name = model_name
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Scaler loaded")
            else:
                print(f"⚠️ Scaler not found, predictions may be inaccurate")
            
            # Load label encoder
            encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"✅ Label encoder loaded ({len(self.label_encoder.classes_)} classes)")
            else:
                print(f"❌ Label encoder not found: {encoder_path}")
                return False
            
            # Load model
            model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            print(f"✅ Model loaded: {model_name}")
            print(f"📊 Features ({len(self.features)}): {self.features}")
            print(f"🏷️ Classes ({len(self.classes)}): {self.classes[:5]}..." if len(self.classes) > 5 else f"🏷️ Classes: {self.classes}")
            
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, features_dict=None, features_array=None):
        """
        Make prediction using loaded model.
        
        Args:
            features_dict: Dictionary of {feature_name: value}
            features_array: List/array of feature values in correct order
        
        Returns:
            dict: {label, class_index, confidence, probabilities}
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Prepare input
        if features_dict is not None:
            # Extract values in correct feature order
            try:
                values = [features_dict[f] for f in self.features]
            except KeyError as e:
                raise ValueError(f"Missing feature: {e}")
            X = np.array([values])
        elif features_array is not None:
            X = np.array([features_array])
        else:
            raise ValueError("Provide either features_dict or features_array")
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Predict
        pred_idx = self.model.predict(X_scaled)[0]
        
        # Get probabilities if available
        probabilities = None
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)[0]
            probabilities = {self.label_encoder.classes_[i]: float(p) for i, p in enumerate(proba)}
            confidence = float(proba.max())
        
        # Decode label
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return {
            'label': label,
            'class_index': int(pred_idx),
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def predict_batch(self, features_list):
        """
        Make predictions for multiple samples.
        
        Args:
            features_list: List of feature arrays
        
        Returns:
            list: List of prediction results
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        X = np.array(features_list)
        
        # Scale
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Predict
        predictions = self.model.predict(X_scaled)
        labels = self.label_encoder.inverse_transform(predictions)
        
        return list(labels)
    
    def get_available_models(self):
        """
        List available model files in models directory.
        
        Returns:
            list: List of model names (without .pkl extension)
        """
        models = []
        for f in os.listdir(self.models_dir):
            if f.endswith('.pkl') and f not in ['scaler.pkl', 'label_encoder.pkl']:
                models.append(f.replace('.pkl', ''))
        return models


# ============================================================
# Interactive CLI Functions
# ============================================================

def get_user_features(features):
    """
    Prompt user to input feature values.
    
    Args:
        features: List of feature names
    
    Returns:
        dict: Feature values
    """
    print("\n📝 Enter feature values:")
    print("-" * 40)
    
    values = {}
    for i, feat in enumerate(features, 1):
        while True:
            try:
                val = input(f"  {i}. {feat}: ")
                values[feat] = float(val)
                break
            except ValueError:
                print("     ⚠️ Please enter a numeric value")
    
    return values


def display_prediction(result):
    """
    Display prediction result nicely.
    
    Args:
        result: Prediction result dict
    """
    print("\n" + "=" * 50)
    print("🎯 PREDICTION RESULT")
    print("=" * 50)
    
    label = result['label']
    confidence = result['confidence']
    
    # Color based on result
    if label == 'BENIGN':
        print(f"   Status: ✅ {label}")
    else:
        print(f"   Status: 🚨 ATTACK DETECTED - {label}")
    
    if confidence is not None:
        print(f"   Confidence: {confidence:.2%}")
    
    # Show top probabilities
    if result['probabilities']:
        print("\n   📊 Top Probabilities:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]
        for cls, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            print(f"      {cls[:25]:25} {prob:6.2%} {bar}")
    
    print("=" * 50)


# ============================================================
# Main CLI
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🛡️  IDS with Machine Learning - CLI Tool")
    print("=" * 60)
    
    # Initialize IDS Model
    ids = IDSModel()
    
    # Show available models
    available = ids.get_available_models()
    print(f"\n📂 Available models: {available}")
    
    # Select model
    print("\nSelect model to load:")
    for i, m in enumerate(available, 1):
        marker = "⭐" if m == 'random_forest' else "  "
        print(f"  {marker} {i}. {m}")
    
    while True:
        sel = input("\nEnter model number (or press Enter for random_forest): ").strip()
        if sel == "":
            model_name = 'random_forest'
            break
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(available):
                model_name = available[idx]
                break
        except ValueError:
            pass
        print("⚠️ Invalid selection, try again")
    
    # Load model
    print(f"\n🔄 Loading {model_name}...")
    if not ids.load(model_name):
        print("❌ Failed to load model. Exiting.")
        return
    
    # Prediction loop
    print("\n" + "=" * 60)
    print("🎯 Ready for predictions!")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("  1. Enter features manually")
        print("  2. Test with sample data")
        print("  3. Switch model")
        print("  4. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            # Manual input
            try:
                features = get_user_features(ids.features)
                result = ids.predict(features_dict=features)
                display_prediction(result)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '2':
            # Test with realistic sample data
            print("\n🧪 Test with sample data:")
            print("  Select sample type:")
            print("  a. Normal traffic (BENIGN-like)")
            print("  b. DDoS-like traffic")
            print("  c. Portscan-like traffic")
            print("  d. DoS Slowloris-like traffic")
            
            sample_choice = input("\n  Enter choice (a/b/c/d): ").strip().lower()
            
            # Define realistic sample values based on CIC-IDS2017 dataset patterns
            samples = {
                'a': {  # BENIGN - Normal traffic patterns
                    "Attempted Category": 0.0,
                    "Bwd Segment Size Avg": 156.5,
                    "Bwd Packet Length Mean": 312.4,
                    "Bwd Packet Length Std": 425.8,
                    "Bwd Packet Length Max": 1460.0,
                    "Subflow Bwd Bytes": 4892.0,
                    "Packet Length Std": 498.2,
                    "Packet Length Mean": 245.6
                },
                'b': {  # DDoS - High volume attack patterns
                    "Attempted Category": 0.0,
                    "Bwd Segment Size Avg": 0.0,
                    "Bwd Packet Length Mean": 0.0,
                    "Bwd Packet Length Std": 0.0,
                    "Bwd Packet Length Max": 0.0,
                    "Subflow Bwd Bytes": 0.0,
                    "Packet Length Std": 28.5,
                    "Packet Length Mean": 74.0
                },
                'c': {  # Portscan - Small packets, reconnaissance
                    "Attempted Category": 0.0,
                    "Bwd Segment Size Avg": 0.0,
                    "Bwd Packet Length Mean": 0.0,
                    "Bwd Packet Length Std": 0.0,
                    "Bwd Packet Length Max": 0.0,
                    "Subflow Bwd Bytes": 0.0,
                    "Packet Length Std": 0.0,
                    "Packet Length Mean": 44.0
                },
                'd': {  # DoS Slowloris - Slow HTTP attack
                    "Attempted Category": 0.0,
                    "Bwd Segment Size Avg": 174.0,
                    "Bwd Packet Length Mean": 174.0,
                    "Bwd Packet Length Std": 0.0,
                    "Bwd Packet Length Max": 174.0,
                    "Subflow Bwd Bytes": 174.0,
                    "Packet Length Std": 87.5,
                    "Packet Length Mean": 119.5
                }
            }
            
            if sample_choice in samples:
                sample = samples[sample_choice]
                sample_names = {'a': 'Normal (BENIGN-like)', 'b': 'DDoS-like', 'c': 'Portscan-like', 'd': 'DoS Slowloris-like'}
                print(f"\n📊 Testing with {sample_names[sample_choice]} traffic pattern...")
                print(f"   Sample values: {sample}")
                try:
                    result = ids.predict(features_dict=sample)
                    display_prediction(result)
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("⚠️ Invalid selection. Using default normal traffic sample.")
                sample = samples['a']
                try:
                    result = ids.predict(features_dict=sample)
                    display_prediction(result)
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif choice == '3':
            # Switch model
            print("\nSelect new model:")
            for i, m in enumerate(available, 1):
                print(f"  {i}. {m}")
            sel = input("Enter number: ").strip()
            try:
                idx = int(sel) - 1
                if 0 <= idx < len(available):
                    ids.load(available[idx])
            except:
                print("⚠️ Invalid selection")
        
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("⚠️ Invalid option")


if __name__ == '__main__':
    main()