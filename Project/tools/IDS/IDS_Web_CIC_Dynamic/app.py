from flask import Flask, render_template, jsonify, request
import os
import pickle
import numpy as np
import pandas as pd
import threading
import logging
import warnings


# For ML training and prediction
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# XGBoost (optional - will be checked at runtime)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

# LightGBM (optional - will be checked at runtime)
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not installed. Install with: pip install lightgbm")

# Balancing methods
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

# ---------------------------
# Logging and warnings setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def recommend_balancing_method(class_counts, total_samples, num_features, num_classes):
    """
    Recommend the best balancing method based on dataset characteristics.
    
    Factors considered:
    1. Imbalance ratio (minority/majority)
    2. Dataset size
    3. Number of features (dimensionality)
    4. Number of classes (binary vs multi-class)
    
    Returns:
        dict: {
            'recommended': str (method name),
            'reason': str (explanation),
            'alternatives': list of str (other good options),
            'analysis': dict (detailed analysis)
        }
    """
    counts = sorted(class_counts.values(), reverse=True)
    majority_count = counts[0]
    minority_count = counts[-1]
    
    # Calculate imbalance ratio (0 to 1, where 1 is perfectly balanced)
    imbalance_ratio = minority_count / majority_count if majority_count > 0 else 0
    
    # Determine imbalance severity
    if imbalance_ratio >= 0.5:
        imbalance_severity = "mild"
    elif imbalance_ratio >= 0.1:
        imbalance_severity = "moderate"
    elif imbalance_ratio >= 0.01:
        imbalance_severity = "severe"
    else:
        imbalance_severity = "extreme"
    
    # Determine dataset size category
    if total_samples < 1000:
        size_category = "small"
    elif total_samples < 10000:
        size_category = "medium"
    elif total_samples < 100000:
        size_category = "large"
    else:
        size_category = "very_large"
    
    # Determine dimensionality
    if num_features < 20:
        dimensionality = "low"
    elif num_features < 100:
        dimensionality = "medium"
    else:
        dimensionality = "high"
    
    # Build analysis info
    analysis = {
        "imbalance_ratio": round(imbalance_ratio, 4),
        "imbalance_severity": imbalance_severity,
        "majority_class_count": majority_count,
        "minority_class_count": minority_count,
        "dataset_size": size_category,
        "total_samples": total_samples,
        "num_features": num_features,
        "dimensionality": dimensionality,
        "num_classes": num_classes,
        "is_multiclass": num_classes > 2
    }
    
    # Decision logic for recommendation
    recommended = "class_weight"
    reason = ""
    alternatives = []
    
    # Check if minority class has too few samples for SMOTE (needs at least 6 for k=5 neighbors)
    minority_too_small = minority_count < 6
    many_classes = num_classes > 10
    
    # Case 0: Minority class too small for SMOTE
    if minority_too_small and imbalance_severity in ["severe", "extreme"]:
        recommended = "class_weight"
        reason = f"⚠️ Smallest class has only {minority_count} samples - too few for SMOTE (needs ≥6). Class Weights is the safest option. Consider removing very small classes using 'Filter Target Values' above."
        alternatives = ["random_undersample"]
    
    # Case 1: Mild imbalance - usually no balancing needed
    elif imbalance_severity == "mild":
        recommended = "none"
        reason = f"The dataset is relatively balanced (ratio: {imbalance_ratio:.2%}). No balancing may be needed, but Class Weights can help slightly."
        alternatives = ["class_weight"]
    
    # Case 2: Many classes (>10) with extreme imbalance - Class Weights is safer
    elif many_classes and imbalance_severity == "extreme":
        recommended = "class_weight"
        reason = f"Dataset has {num_classes} classes with extreme imbalance. Class Weights is safest - SMOTE may fail on very small classes. Consider filtering out classes with <10 samples first."
        alternatives = ["smote_enn", "random_undersample"]
    
    # Case 3: Multi-class + Very Large + Severe/Extreme - needs special handling
    elif num_classes > 2 and size_category == "very_large" and imbalance_severity in ["severe", "extreme"]:
        recommended = "smote_enn"
        reason = f"Large multi-class dataset ({num_classes} classes, {total_samples:,} samples) with {imbalance_severity} imbalance ({imbalance_ratio:.2%}). SMOTE-ENN handles multi-class well and cleans noisy synthetic samples."
        alternatives = ["class_weight", "smote_tomek", "adasyn"]
    
    # Case 3: Multi-class + Large + Severe/Extreme
    elif num_classes > 2 and size_category == "large" and imbalance_severity in ["severe", "extreme"]:
        recommended = "smote"
        reason = f"Multi-class dataset ({num_classes} classes) with {imbalance_severity} imbalance. SMOTE works well for generating balanced multi-class samples."
        alternatives = ["class_weight", "smote_enn", "adasyn"]
    
    # Case 4: Multi-class with moderate imbalance or smaller datasets
    elif num_classes > 2:
        recommended = "class_weight"
        reason = f"For multi-class problems ({num_classes} classes) with {imbalance_severity} imbalance, Class Weights is fast and safe. It adjusts model weights without creating synthetic samples."
        if size_category in ["large", "very_large"]:
            alternatives = ["smote", "adasyn"]
        else:
            alternatives = ["smote"]
    
    # Case 5: Small dataset (binary)
    elif size_category == "small":
        recommended = "class_weight"
        reason = f"With a small dataset ({total_samples} samples), Class Weights avoids overfitting that synthetic samples might cause."
        alternatives = ["smote"]  # SMOTE can work but with caution
    
    # Case 6: Very large dataset (binary)
    elif size_category == "very_large":
        if imbalance_severity in ["severe", "extreme"]:
            recommended = "smote_enn"
            reason = f"Large dataset with {imbalance_severity} imbalance ({imbalance_ratio:.2%}). SMOTE-ENN cleans noisy samples after oversampling for better results."
            alternatives = ["class_weight", "random_undersample", "smote_tomek"]
        else:
            recommended = "class_weight"
            reason = "Large dataset with moderate imbalance. Class Weights is fast and effective."
            alternatives = ["smote", "random_undersample"]
    
    # Case 7: High dimensionality
    elif dimensionality == "high":
        recommended = "class_weight"
        reason = f"High-dimensional data ({num_features} features) works better with Class Weights. SMOTE may create unrealistic synthetic samples in high dimensions."
        alternatives = ["random_undersample"]
    
    # Case 8: Moderate imbalance with medium/large dataset
    elif imbalance_severity == "moderate":
        if size_category == "medium":
            recommended = "smote"
            reason = f"Moderate imbalance ({imbalance_ratio:.2%}) with medium-sized dataset. SMOTE creates synthetic minority samples effectively."
            alternatives = ["class_weight", "adasyn"]
        else:  # large
            recommended = "class_weight"
            reason = "Moderate imbalance with large dataset. Class Weights is efficient and often sufficient."
            alternatives = ["smote", "smote_enn"]
    
    # Case 7: Severe imbalance
    elif imbalance_severity == "severe":
        if dimensionality == "low":
            recommended = "borderline_smote"
            reason = f"Severe imbalance ({imbalance_ratio:.2%}) with low dimensionality. Borderline-SMOTE focuses on decision boundary samples."
            alternatives = ["smote_enn", "adasyn", "class_weight"]
        else:
            recommended = "smote_enn"
            reason = f"Severe imbalance ({imbalance_ratio:.2%}). SMOTE-ENN combines oversampling with cleaning for better quality samples."
            alternatives = ["class_weight", "borderline_smote"]
    
    # Case 8: Extreme imbalance
    elif imbalance_severity == "extreme":
        recommended = "smote_enn"
        reason = f"Extreme imbalance (ratio: {imbalance_ratio:.4%}). SMOTE-ENN or combined methods are needed. Consider also collecting more minority samples if possible."
        alternatives = ["smote_tomek", "class_weight", "adasyn"]
    
    # Default fallback
    else:
        recommended = "class_weight"
        reason = "Class Weights is a safe, fast choice that works well for most IDS datasets."
        alternatives = ["smote"]
    
    return {
        "recommended": recommended,
        "reason": reason,
        "alternatives": alternatives,
        "analysis": analysis
    }

# ---------------------------
# Flask app creation
# ---------------------------
# Set template_folder and static_folder as needed for the ML part.
app = Flask(__name__, template_folder='templates', static_folder='static')

# ---------------------------
# Global variables for ML functionality
# ---------------------------
data_store = {       # Store uploaded dataset.
    "df": None,           # Original dataset
    "df_balanced": None,  # SMOTE-balanced dataset
    "smote_applied": False # Flag to track if SMOTE has been applied
}
model_config = {}     # Stores configuration and trained models.
loaded_model_config = {}  # For loaded models and dataset.

# Directories for file uploads and exported models.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
uploads_dir = os.path.join(BASE_DIR, "uploads")
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Constants for large file processing.
CHUNK_THRESHOLD = 100 * 1024 * 1024      # 100 MB
LARGE_FILE_THRESHOLD = 500 * 1024 * 1024   # 500 MB
DASK_BLOCKSIZE = 10e6  # 10 MB
LARGE_DATASET_THRESHOLD = 100000  # Used to decide algorithm selection.

# ---------------------------
# ML API Endpoints
# ---------------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "dataset" not in request.files:
        return jsonify({"error": "No files uploaded."}), 400
    files = request.files.getlist("dataset")
    if len(files) == 0 or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected."}), 400

    # Reset SMOTE status when uploading new dataset
    data_store["df_balanced"] = None
    data_store["smote_applied"] = False
    
    dfs = []
    reference_columns = None

    for file in files:
        if file.filename.lower().endswith((".csv", ".txt")):
            try:
                save_path = os.path.join(uploads_dir, file.filename)
                if not os.path.exists(save_path):
                    file.save(save_path)
                else:
                    logging.info(f"File {file.filename} already exists. Skipping re-save.")

                file_size = os.path.getsize(save_path)
                logging.info(f"Processing file {file.filename} of size {file_size/1024/1024:.2f} MB")

                if file_size > LARGE_FILE_THRESHOLD:
                    logging.info(f"Using Dask to process large file {file.filename}")
                    df = dd.read_csv(save_path, blocksize=DASK_BLOCKSIZE).compute()
                elif file_size > CHUNK_THRESHOLD:
                    chunks = pd.read_csv(save_path, sep=None, engine="python", chunksize=100000)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(save_path, sep=None, engine="python")

                if reference_columns is None:
                    reference_columns = df.columns.tolist()
                else:
                    if df.columns.tolist() != reference_columns:
                        return jsonify({"error": f"Column mismatch in {file.filename}. All files must have identical columns."}), 400

                dfs.append(df)

            except Exception as e:
                logging.exception("Error reading file")
                return jsonify({"error": f"Error reading {file.filename}: {str(e)}"}), 500
        else:
            return jsonify({"error": f"Unsupported file type: {file.filename}"}), 400

    if not dfs:
        return jsonify({"error": "No valid files processed."}), 400

    try:
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        
        # Remove constant features (columns with only one unique value)
        constant_features = [col for col in combined_df.columns if combined_df[col].nunique() <= 1]
        if constant_features:
            logging.info(f"Removing {len(constant_features)} constant features: {constant_features}")
            combined_df = combined_df.drop(columns=constant_features)
        
        # Remove duplicated features with '.1' in their names using regex
        duplicate_features = [col for col in combined_df.columns if col.endswith('.1')]
        if duplicate_features:
            logging.info(f"Removing {len(duplicate_features)} duplicated features with '.1': {duplicate_features}")
            combined_df = combined_df.drop(columns=duplicate_features)
        
        # Store the cleaned dataframe
        data_store["df"] = combined_df
        
        # Prepare message about removed features
        removed_features_msg = ""
        if constant_features or duplicate_features:
            removed_features_msg = f" Removed {len(constant_features)} constant features and {len(duplicate_features)} duplicated features."
        
        return jsonify({
            "columns": combined_df.columns.tolist(),
            "message": f"Successfully processed {len(dfs)} file(s). Total rows: {len(combined_df)}.{removed_features_msg}"
        })
    except Exception as e:
        logging.exception("Error combining files")
        return jsonify({"error": f"Error combining files: {str(e)}"}), 500

@app.route("/preview_data", methods=["POST"])
def preview_data():
    """Return preview of uploaded dataset (first N rows)"""
    req = request.get_json()
    num_rows = req.get("num_rows", 10)  # Default to 10 rows
    
    if data_store["df"] is None:
        return jsonify({"error": "No dataset uploaded."}), 400
    
    df = data_store["df"]
    
    # Get preview data
    preview_df = df.head(num_rows)
    
    # Convert to JSON-friendly format
    preview_data = preview_df.to_dict(orient='records')
    columns = df.columns.tolist()
    
    # Get basic statistics
    stats = {
        "total_rows": len(df),
        "total_columns": len(columns),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=['object']).columns)
    }
    
    return jsonify({
        "preview": preview_data,
        "columns": columns,
        "stats": stats
    })

@app.route("/filter_target_values", methods=["POST"])
def filter_target_values():
    """Remove selected values from the target column (remove rows with those values)"""
    req = request.get_json()
    target_column = req.get("target_column")
    values_to_remove = req.get("values_to_remove", [])
    
    if data_store["df"] is None:
        return jsonify({"error": "No dataset uploaded."}), 400
    
    if not target_column:
        return jsonify({"error": "Target column not specified."}), 400
    
    if not values_to_remove:
        return jsonify({"error": "No values selected for removal."}), 400
    
    df = data_store["df"]
    
    if target_column not in df.columns:
        return jsonify({"error": f"Target column '{target_column}' not found in dataset."}), 400
    
    # Get current target values
    y = df[target_column].copy()
    if pd.api.types.is_object_dtype(y):
        y = y.str.strip()
    
    current_values = y.unique().tolist()
    
    # Validate values exist
    invalid_values = [v for v in values_to_remove if v not in current_values]
    if invalid_values:
        return jsonify({"error": f"Values not found in target: {', '.join(map(str, invalid_values))}"}), 400
    
    # Check if trying to remove all values
    remaining_values = [v for v in current_values if v not in values_to_remove]
    if len(remaining_values) < 1:
        return jsonify({"error": "Cannot remove all target values. At least 1 class must remain."}), 400
    
    try:
        original_count = len(df)
        
        # Remove rows where target value is in the removal list
        if pd.api.types.is_object_dtype(df[target_column]):
            mask = ~df[target_column].str.strip().isin(values_to_remove)
        else:
            mask = ~df[target_column].isin(values_to_remove)
        
        data_store["df"] = df[mask].reset_index(drop=True)
        
        # Reset balanced dataset if SMOTE was applied
        if data_store["smote_applied"]:
            data_store["smote_applied"] = False
            data_store["df_balanced"] = None
        
        new_count = len(data_store["df"])
        removed_rows = original_count - new_count
        
        # Get updated class distribution
        new_y = data_store["df"][target_column]
        if pd.api.types.is_object_dtype(new_y):
            new_y = new_y.str.strip()
        new_class_counts = new_y.value_counts().to_dict()
        
        logging.info(f"Removed {len(values_to_remove)} target values ({removed_rows} rows): {values_to_remove}")
        
        return jsonify({
            "success": True,
            "removed_values": values_to_remove,
            "removed_rows": removed_rows,
            "remaining_values": remaining_values,
            "remaining_count": len(remaining_values),
            "total_rows": new_count,
            "new_class_counts": new_class_counts,
            "message": f"Successfully removed {len(values_to_remove)} class(es) ({removed_rows:,} rows). {len(remaining_values)} classes remaining with {new_count:,} total rows."
        })
    except Exception as e:
        logging.error(f"Error filtering target values: {str(e)}")
        return jsonify({"error": f"Error filtering target values: {str(e)}"}), 500

@app.route("/get_target_values", methods=["POST"])
def get_target_values():
    """Get unique values in the target column"""
    req = request.get_json()
    target_column = req.get("target_column")
    
    if data_store["df"] is None:
        return jsonify({"error": "No dataset uploaded."}), 400
    
    if not target_column or target_column not in data_store["df"].columns:
        return jsonify({"error": "Target column not specified or not found."}), 400
    
    y = data_store["df"][target_column].copy()
    if pd.api.types.is_object_dtype(y):
        y = y.str.strip()
    
    # Get value counts
    value_counts = y.value_counts().to_dict()
    values = list(value_counts.keys())
    
    return jsonify({
        "values": values,
        "value_counts": value_counts,
        "count": len(values)
    })

@app.route("/analyze_class_distribution", methods=["POST"])
def analyze_class_distribution():
    req = request.get_json()
    target = req.get("target")
    if data_store["df"] is None or not target:
        return jsonify({"error": "Dataset not uploaded or target not specified."}), 400

    # Use the balanced dataset if SMOTE has been applied, otherwise use the original
    using_balanced = False
    if data_store["smote_applied"] and data_store["df_balanced"] is not None:
        df = data_store["df_balanced"]
        using_balanced = True
        logging.info("Using SMOTE-balanced dataset for class distribution analysis")
    else:
        df = data_store["df"]
        logging.info("Using original dataset for class distribution analysis")

    if target not in df.columns:
        return jsonify({"error": f"Target column '{target}' not found in dataset."}), 400

    y = df[target].copy()
    if pd.api.types.is_object_dtype(y):
        y = y.str.strip()

    # Calculate class distribution
    class_counts = y.value_counts().to_dict()
    total_samples = len(y)
    class_percentages = {k: (v / total_samples) * 100 for k, v in class_counts.items()}
    
    # Sort by count (descending)
    sorted_classes = sorted(class_percentages.items(), key=lambda x: class_counts[x[0]], reverse=True)
    
    # Calculate number of features (excluding target column)
    num_features = len(df.columns) - 1
    num_classes = len(class_counts)
    
    # Get balancing method recommendation
    recommendation = recommend_balancing_method(
        class_counts=class_counts,
        total_samples=total_samples,
        num_features=num_features,
        num_classes=num_classes
    )
    
    return jsonify({
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "sorted_classes": sorted_classes,
        "total_samples": total_samples,
        "using_balanced_dataset": using_balanced,
        "recommendation": recommendation
    })

@app.route("/reset_smote", methods=["POST"])
def reset_smote():
    """Reset the SMOTE status and revert to using the original dataset"""
    try:
        data_store["smote_applied"] = False
        data_store["df_balanced"] = None
        logging.info("SMOTE status reset - reverting to original dataset")
        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Error resetting SMOTE status: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500 # Return the error message
@app.route("/apply_smote", methods=["POST"])
def apply_smote_endpoint():
    """
    Apply SMOTE to preview balanced class distribution.
    
    NOTE: This endpoint is for DATA PREVIEW/ANALYSIS ONLY.
    The actual model training (/configure endpoint) applies SMOTE only to 
    training data to prevent data leakage. This endpoint shows how the 
    balanced dataset would look like for exploratory purposes.
    """
    req = request.get_json()
    target = req.get("target")
    # Get memory-efficient options if provided
    downsample_ratio = req.get("downsample_ratio", 1.0)  # Default: no downsampling
    max_per_class = req.get("max_per_class", None)      # Default: no max limit
    memory_efficient = req.get("memory_efficient", False) # Default: standard SMOTE
    
    # Progress tracking variables
    progress_steps = 0
    total_steps = 6  # Total number of major steps in SMOTE process
    
    def log_progress(message, step_increment=1):
        nonlocal progress_steps
        progress_steps += step_increment
        percentage = min(round((progress_steps / total_steps) * 100), 100)
        logging.info(f"[PROGRESS {percentage}%] {message}")
    
    log_progress("Starting SMOTE process", 0)  # Initialize with 0% progress
    
    if data_store["df"] is None or not target:
        return jsonify({"error": "Dataset not uploaded or target not specified."}), 400
    
    # Check if SMOTE is already applied
    if data_store["smote_applied"] and data_store["df_balanced"] is not None:
        log_progress("SMOTE already applied, skipping process", total_steps)  # Mark as 100% complete
        return jsonify({
            "message": "SMOTE has already been applied to the dataset.",
            "smote_applied": True
        })
    
    df = data_store["df"]
    if target not in df.columns:
        return jsonify({"error": f"Target column '{target}' not found in dataset."}), 400
    
    log_progress("Loading and preparing dataset")
    
    # Get target column
    y = df[target].copy()
    if pd.api.types.is_object_dtype(y):
        y = y.str.strip()
    
    # Check if target has enough unique values for SMOTE
    unique_values = np.unique(y)
    if len(unique_values) < 2:
        return jsonify({"error": "SMOTE requires at least 2 classes in the target variable."}), 400
    
    # Get all features except target
    features = [col for col in df.columns if col != target]
    X = df[features].copy()
    
    # Clean and prepare data
    log_progress("Cleaning and preparing features")
    # Handle categorical features
    categorical_features = []
    numeric_features = []
    le_dict = {}
    
    for col in features:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_features.append(col)
            # Handle infinities and NaNs
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].mean() if not X[col].isna().all() else 0)
        else:
            categorical_features.append(col)
            # Clean strings
            if pd.api.types.is_object_dtype(X[col]):
                X[col] = X[col].astype(str).str.strip()
            # Handle missing values
            mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else "unknown"
            X[col] = X[col].fillna(mode_val)
            # Encode categorical features
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
    
    # Encode target if categorical
    target_enc = None
    if pd.api.types.is_object_dtype(y):
        target_enc = LabelEncoder()
        y = target_enc.fit_transform(y.astype(str))
    
    # Ensure y is properly encoded as integers before using np.bincount
    # Convert to numpy array of integers for compatibility with np.bincount
    if isinstance(y, pd.Series):
        # Handle PyArrow string types by converting to standard Python types first
        if 'pyarrow' in str(y.dtype).lower():
            y = y.astype(str)
        y = y.values
    
    # Make sure y is already encoded as integers before converting to numpy array
    # This prevents errors when trying to convert string values like 'BENIGN' to int
    try:
        is_integer = np.issubdtype(y.dtype, np.integer)
    except TypeError:
        # Handle PyArrow string types or other incompatible types
        is_integer = False
        
    if target_enc is None and not is_integer:
        # If we reach here, y contains non-integer values but wasn't encoded
        # This is likely a categorical variable that needs encoding
        target_enc = LabelEncoder()
        y = target_enc.fit_transform(y.astype(str))
    
    # Now it's safe to convert to int array
    y = np.array(y, dtype=int)
    
    try:
        logging.info(f"Applying SMOTE to balance classes")
        # Log class distribution before SMOTE
        class_counts = np.bincount(y)        
        logging.info(f"Before SMOTE: {class_counts}")
        
        log_progress("Analyzing dataset size and class distribution")
        
        # Check if dataset is too large and needs downsampling
        total_samples = len(y)
        logging.info(f"Dataset size: {total_samples} samples, {X.shape[1]} features")
        
        # Find the second highest class count to use as target for majority class downsampling
        sorted_counts = np.sort(class_counts)[::-1]  # Sort in descending order
        if len(sorted_counts) >= 2:
            # Get the second highest count
            second_highest_count = sorted_counts[1]
            logging.info(f"Second highest class count: {second_highest_count}")
            
            # Find the majority class (class with highest count)
            majority_class = np.argmax(class_counts)
            majority_count = class_counts[majority_class]
            logging.info(f"Majority class: {majority_class} with count: {majority_count}")
            
            # Downsample the majority class to match the second highest class count
            X_downsampled = []
            y_downsampled = []
            
            # Process each class separately
            for class_label in np.unique(y):
                # Get indices for this class
                indices = np.where(y == class_label)[0]
                class_size = len(indices)
                
                # Calculate how many samples to keep
                if class_label == majority_class:
                    # For majority class, downsample to second highest count
                    samples_to_keep = second_highest_count
                    logging.info(f"Downsampling majority class {class_label} from {class_size} to {samples_to_keep}")
                else:
                    # For other classes, keep all samples
                    samples_to_keep = class_size
                
                # Randomly select indices to keep if downsampling is needed
                if samples_to_keep < class_size:
                    np.random.seed(42)  # For reproducibility
                    keep_indices = np.random.choice(indices, samples_to_keep, replace=False)
                else:
                    keep_indices = indices
                
                # Add to downsampled data
                X_downsampled.append(X.iloc[keep_indices])
                y_downsampled.extend([class_label] * len(keep_indices))
            
            # Combine downsampled data
            X = pd.concat(X_downsampled, axis=0).reset_index(drop=True)
            y = np.array(y_downsampled)
            
            logging.info(f"After downsampling majority class: {len(y)} samples")
            logging.info(f"Class distribution after downsampling: {np.bincount(y)}")
            
            log_progress("Completed majority class downsampling")
        else:
            logging.info("Not enough classes to perform majority class downsampling")
            
            # Apply regular downsampling if requested or if dataset is very large
            if downsample_ratio < 1.0 or max_per_class is not None or memory_efficient:
                logging.info("Applying memory-efficient processing")
                
                # Downsample classes if needed
                X_downsampled = []
                y_downsampled = []
                
                # Process each class separately
                for class_label in np.unique(y):
                    # Get indices for this class
                    indices = np.where(y == class_label)[0]
                    class_size = len(indices)
                    
                    # Calculate how many samples to keep
                    if max_per_class is not None:
                        # Cap at max_per_class
                        samples_to_keep = min(class_size, max_per_class)
                    else:
                        # Apply downsample_ratio
                        samples_to_keep = max(int(class_size * downsample_ratio), 10)  # Keep at least 10 samples
                    
                    # Randomly select indices to keep
                    if samples_to_keep < class_size:
                        np.random.seed(42)  # For reproducibility
                        keep_indices = np.random.choice(indices, samples_to_keep, replace=False)
                    else:
                        keep_indices = indices
                    
                    # Add to downsampled data
                    X_downsampled.append(X.iloc[keep_indices])
                    y_downsampled.extend([class_label] * len(keep_indices))
                
                # Combine downsampled data
                X = pd.concat(X_downsampled, axis=0).reset_index(drop=True)
                y = np.array(y_downsampled)
                
                logging.info(f"After downsampling: {len(y)} samples")
                logging.info(f"Class distribution after downsampling: {np.bincount(y)}")
        
        # Get class counts to determine appropriate k value for SMOTE
        min_samples = min(count for count in np.bincount(y) if count > 0)
        
        # Determine k value (number of neighbors) based on smallest class size
        # k must be less than the number of samples in the smallest class
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        logging.info(f"Using k_neighbors={k_neighbors} for SMOTE based on smallest class size of {min_samples}")
        
        # Apply SMOTE with adjusted k_neighbors parameter
        try:
            log_progress("Applying SMOTE algorithm")
            # Try with standard SMOTE first
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy='auto')
            X_balanced, y_balanced = smote.fit_resample(X, y)
        except MemoryError as me:
            # If memory error occurs, try with more aggressive downsampling
            logging.warning(f"Memory error during SMOTE: {str(me)}. Trying with more aggressive downsampling.")
            
            # More aggressive downsampling of majority classes
            X_downsampled = []
            y_downsampled = []
            
            # Find minority class size
            min_class_size = min(np.bincount(y)[np.nonzero(np.bincount(y))[0]])
            
            # Process each class separately
            for class_label in np.unique(y):
                # Get indices for this class
                indices = np.where(y == class_label)[0]
                class_size = len(indices)
                
                # For majority classes, limit to 2x the minority class size
                if class_size > min_class_size:
                    samples_to_keep = min(class_size, 2 * min_class_size)
                    np.random.seed(42)
                    keep_indices = np.random.choice(indices, samples_to_keep, replace=False)
                else:
                    keep_indices = indices
                
                # Add to downsampled data
                X_downsampled.append(X.iloc[keep_indices])
                y_downsampled.extend([class_label] * len(keep_indices))
            
            # Combine downsampled data
            X = pd.concat(X_downsampled, axis=0).reset_index(drop=True)
            y = np.array(y_downsampled)
            
            logging.info(f"After aggressive downsampling: {len(y)} samples")
            logging.info(f"Class distribution after aggressive downsampling: {np.bincount(y)}")
            
            # Try SMOTE again with downsampled data
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        
        logging.info(f"After SMOTE: {np.bincount(y_balanced)}")
        
        log_progress("SMOTE completed, converting results back to DataFrame")
        
        # Convert back to DataFrame
        balanced_df = pd.DataFrame(X_balanced, columns=features)
        
        # Convert categorical features back to original representation
        for col in categorical_features:
            balanced_df[col] = le_dict[col].inverse_transform(balanced_df[col].astype(int))
        
        # Convert target back if needed
        if target_enc is not None:
            y_balanced_series = pd.Series(target_enc.inverse_transform(y_balanced))
        else:
            y_balanced_series = pd.Series(y_balanced)
        
        # Add target column to balanced DataFrame
        balanced_df[target] = y_balanced_series
        
        # Store the balanced dataset
        data_store["df_balanced"] = balanced_df
        data_store["smote_applied"] = True
        
        log_progress("Finalizing and storing balanced dataset", 2)  # Final step counts as 2 to reach 100%
        
        # Calculate new class distribution
        new_class_counts = y_balanced_series.value_counts().to_dict()
        total_samples = len(y_balanced_series)
        new_class_percentages = {k: (v / total_samples) * 100 for k, v in new_class_counts.items()}
        
        # Sort by count (descending)
        new_sorted_classes = sorted(new_class_percentages.items(), key=lambda x: new_class_counts[x[0]], reverse=True)
        
        return jsonify({
            "message": "SMOTE successfully applied to balance the dataset.",
            "smote_applied": True,
            "original_shape": df.shape,
            "balanced_shape": balanced_df.shape,
            "class_counts": new_class_counts,
            "class_percentages": new_class_percentages,
            "sorted_classes": new_sorted_classes,
            "memory_efficient_applied": memory_efficient or downsample_ratio < 1.0 or max_per_class is not None
        })
    
    except Exception as e:
        logging.exception(f"Error applying SMOTE: {str(e)}")
        return jsonify({
            "error": f"Error applying SMOTE: {str(e)}",
            "smote_applied": False,
            "suggestion": "Try using memory_efficient=true, reducing downsample_ratio, or setting max_per_class to limit dataset size."
        }), 500

@app.route("/recommend_features", methods=["POST"])
def recommend_features():
    req = request.get_json()
    target = req.get("target")
    k_val = req.get("k", 10)  # default is 10 features
    
    # Check if we have any dataset available
    if data_store["df"] is None or not target:
        return jsonify({"error": "Dataset not uploaded or target not specified."}), 400
    
    # Use the balanced dataset if SMOTE has been applied, otherwise use the original
    if data_store["smote_applied"] and data_store["df_balanced"] is not None:
        df = data_store["df_balanced"]
        logging.info("Using SMOTE-balanced dataset for feature recommendation")
    else:
        df = data_store["df"]
        logging.info("Using original dataset for feature recommendation")
    
    if target not in df.columns:
        return jsonify({"error": f"Target column '{target}' not found in dataset."}), 400

    features = [col for col in df.columns if col != target]
    X = df[features].copy()
    y = df[target].copy()

    # Clean string columns
    obj_cols = X.select_dtypes(include=['object']).columns
    if not obj_cols.empty:
        X[obj_cols] = X[obj_cols].apply(lambda col: col.str.strip())
    if pd.api.types.is_object_dtype(y):
        y = y.str.strip()

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].where(np.isfinite(X[col]), np.nan)
            X[col] = X[col].fillna(X[col].mean())

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else "unknown"
            X[col] = X[col].fillna(mode_val)
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    try:
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        non_constant_cols = [col for col in X.columns if col not in constant_cols]
        feature_scores = {}

        if non_constant_cols:
            k_adjusted = min(k_val, len(non_constant_cols))
            selector = SelectKBest(score_func=f_classif, k=k_adjusted)
            selector.fit(X[non_constant_cols], y)
            scores = selector.scores_
            for col, score in zip(non_constant_cols, scores):
                feature_scores[col] = 0 if np.isnan(score) or np.isinf(score) else score

        for col in constant_cols:
            feature_scores[col] = 0

        recommended = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
        return jsonify({
            "recommended_features": recommended[:k_val],
            "all_feature_scores": feature_scores,
            "using_balanced_dataset": data_store["smote_applied"] and data_store["df_balanced"] is not None
        })
    except Exception as e:
        logging.exception("Error during feature selection")
        return jsonify({"error": f"Error during feature selection: {str(e)}"}), 500

@app.route("/get_available_models", methods=["GET"])
def get_available_models():
    """Return list of available ML models for training"""
    models = [
        {"id": "random_forest", "name": "Random Forest", "description": "Ensemble of decision trees, good for most cases", "default": True},
        {"id": "logistic_regression", "name": "Logistic Regression", "description": "Fast and interpretable linear model", "default": True},
        {"id": "decision_tree", "name": "Decision Tree", "description": "Simple and interpretable tree-based model", "default": False},
        {"id": "knn", "name": "K-Nearest Neighbors", "description": "Instance-based learning, good for smaller datasets", "default": False},
        {"id": "naive_bayes", "name": "Naive Bayes", "description": "Fast probabilistic classifier", "default": False},
        {"id": "svm", "name": "Support Vector Machine", "description": "Effective for high-dimensional data (slower on large datasets)", "default": False},
        {"id": "gradient_boosting", "name": "Gradient Boosting", "description": "Powerful boosting algorithm, slower training", "default": False},
        {"id": "adaboost", "name": "AdaBoost", "description": "Adaptive boosting, focuses on hard examples", "default": False},
        {"id": "extra_trees", "name": "Extra Trees", "description": "Similar to Random Forest with more randomness", "default": False},
        {"id": "xgboost", "name": "XGBoost", "description": "Optimized gradient boosting, very powerful", "available": XGBOOST_AVAILABLE, "default": False},
        {"id": "lightgbm", "name": "LightGBM", "description": "Fast gradient boosting by Microsoft", "available": LIGHTGBM_AVAILABLE, "default": False}
    ]
    
    # Filter out unavailable models
    available_models = []
    for model in models:
        if model.get("available", True):
            available_models.append(model)
    
    return jsonify({"models": available_models})

@app.route("/configure", methods=["POST"])
def configure():
    req = request.get_json()
    features = req.get("input_features")
    target = req.get("target")
    balancing_method = req.get("balancing_method", "none")  # New: multiple balancing options
    apply_smote = req.get("apply_smote", False)  # Legacy support
    test_size = req.get("test_size", 0.30)  # Default 30% for validation+test (15% each)
    selected_models = req.get("selected_models", ["random_forest", "logistic_regression"])  # User selected models
    
    # Legacy support: if apply_smote is True and no balancing_method specified
    if apply_smote and balancing_method == "none":
        balancing_method = "smote"
    
    # Validate test_size (should be between 0.1 and 0.5)
    try:
        test_size = float(test_size)
        test_size = max(0.1, min(0.5, test_size))
    except (ValueError, TypeError):
        test_size = 0.30
    
    # Validate selected_models - ensure at least one model is selected
    if not selected_models or len(selected_models) == 0:
        selected_models = ["random_forest", "logistic_regression"]
    
    # Progress tracking variables
    progress_steps = 0
    total_steps = 10  # Total number of major steps in model training process
    
    def log_progress(message, step_increment=1):
        nonlocal progress_steps
        progress_steps += step_increment
        percentage = min(round((progress_steps / total_steps) * 100), 100)
        logging.info(f"[PROGRESS {percentage}%] {message}")
    
    log_progress("Starting model configuration and training", 0)  # Initialize with 0% progress
    
    if data_store["df"] is None or not features or not target or target in features:
        return jsonify({"error": "Invalid configuration."}), 400

    # Use the balanced dataset if SMOTE has been applied, otherwise use the original
    if data_store["smote_applied"] and data_store["df_balanced"] is not None:
        df = data_store["df_balanced"]
        logging.info("Using SMOTE-balanced dataset for model training")
    else:
        df = data_store["df"]
        logging.info("Using original dataset for model training")
        
    for col in features + [target]:
        if col not in df.columns:
            return jsonify({"error": f"Column '{col}' missing."}), 400

    log_progress("Loading dataset and extracting features")
    
    X = df[features].copy()
    y = df[target].copy()

    obj_cols = X.select_dtypes(include=['object']).columns
    if not obj_cols.empty:
        X[obj_cols] = X[obj_cols].apply(lambda col: col.str.strip())
    if pd.api.types.is_object_dtype(y):
        y = y.str.strip()

    numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = [col for col in features if col not in numeric_features]

    for col in numeric_features:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        X[col] = X[col].fillna(0)

    for col in categorical_features:
        if X[col].isna().any():
            mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else "unknown"
            X[col] = X[col].fillna(mode_val)

    if pd.api.types.is_numeric_dtype(y):
        target_type = "numeric"
        target_enc = None
    else:
        target_type = "categorical"
        target_enc = LabelEncoder()
        y = target_enc.fit_transform(y.astype(str))

    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    log_progress("Splitting dataset into train/validation/test sets")
    
    # IMPORTANT: Split data BEFORE applying SMOTE to prevent data leakage
    # Using dynamic test_size from slider (default: 70% train, 15% validation, 15% test)
    train_size = round((1 - test_size) * 100)
    val_test_each = round((test_size / 2) * 100)
    logging.info(f"Split ratio - Train: {train_size}%, Val: {val_test_each}%, Test: {val_test_each}%")
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    logging.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Balancing should only be applied to training data to prevent data leakage
    balancing_applied = False
    balancing_method_used = "none"
    use_class_weight = False  # Special flag for class_weight method
    
    log_progress("Checking balancing method and preparing training data")
    
    if balancing_method and balancing_method != "none":
        try:
            log_progress(f"Applying {balancing_method} to balance TRAINING DATA ONLY (preventing data leakage)")
            logging.info(f"Before balancing on training data: {np.bincount(y_train)}")
            
            # Determine appropriate k_neighbors based on smallest class size in training data
            min_samples = min(count for count in np.bincount(y_train) if count > 0)
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            logging.info(f"Using k_neighbors={k_neighbors} for sampling methods")
            
            if balancing_method == "class_weight":
                # Class weight is applied to models, not data
                use_class_weight = True
                balancing_applied = True
                balancing_method_used = "class_weight"
                logging.info("Class weights will be applied to models during training")
                
            elif balancing_method == "smote":
                sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                balancing_applied = True
                balancing_method_used = "smote"
                
            elif balancing_method == "smote_enn":
                sampler = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                balancing_applied = True
                balancing_method_used = "smote_enn"
                
            elif balancing_method == "adasyn":
                sampler = ADASYN(random_state=42, n_neighbors=k_neighbors)
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                balancing_applied = True
                balancing_method_used = "adasyn"
                
            elif balancing_method == "borderline_smote":
                sampler = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                balancing_applied = True
                balancing_method_used = "borderline_smote"
                
            elif balancing_method == "random_undersample":
                sampler = RandomUnderSampler(random_state=42)
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                balancing_applied = True
                balancing_method_used = "random_undersample"
                
            elif balancing_method == "tomek":
                sampler = TomekLinks()
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                balancing_applied = True
                balancing_method_used = "tomek"
                
            elif balancing_method == "smote_tomek":
                sampler = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                balancing_applied = True
                balancing_method_used = "smote_tomek"
            
            if balancing_method != "class_weight":
                logging.info(f"After {balancing_method} on training data: {np.bincount(y_train)}")
                logging.info(f"{balancing_method} successfully applied to TRAINING DATA ONLY - validation and test sets remain unchanged")
            
        except Exception as e:
            logging.warning(f"{balancing_method} could not be applied: {str(e)}")
            logging.warning("Continuing without balancing")
            balancing_applied = False
            balancing_method_used = "none"
    else:
        logging.info("No balancing method requested")
    
    # Add balancing status to the model configuration
    model_config["balancing_applied"] = balancing_applied
    model_config["balancing_method"] = balancing_method_used
    model_config["smote_applied"] = balancing_applied  # Legacy support

    log_progress("Scaling features")
    
    # Use RobustScaler - better for network traffic data which often has outliers
    scaler = RobustScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        logging.exception("Error scaling data")
        return jsonify({"error": f"Error scaling data: {str(e)}"}), 400

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    log_progress("Initializing machine learning models")
    
    # Apply class_weight if selected
    class_weight_param = 'balanced' if use_class_weight else None
    
    # Define all available models with their configurations
    # (model_key, model_instance, use_scaling)
    available_models = {}
    
    # Logistic Regression
    if "logistic_regression" in selected_models:
        available_models["logistic_regression"] = {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight_param),
            "use_scaling": True
        }
    
    # Random Forest
    if "random_forest" in selected_models:
        available_models["random_forest"] = {
            "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weight_param),
            "use_scaling": False
        }
    
    # SVM (only for smaller datasets)
    if "svm" in selected_models:
        if len(df) <= LARGE_DATASET_THRESHOLD:
            available_models["svm"] = {
                "model": LinearSVC(random_state=42, max_iter=1000),
                "use_scaling": True
            }
        else:
            logging.warning("SVM skipped - dataset too large. Consider using other models.")
    
    # KNN
    if "knn" in selected_models:
        available_models["knn"] = {
            "model": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "use_scaling": True
        }
    
    # Decision Tree
    if "decision_tree" in selected_models:
        available_models["decision_tree"] = {
            "model": DecisionTreeClassifier(random_state=42, class_weight=class_weight_param),
            "use_scaling": False
        }
    
    # Naive Bayes
    if "naive_bayes" in selected_models:
        available_models["naive_bayes"] = {
            "model": GaussianNB(),
            "use_scaling": True
        }
    
    # Gradient Boosting
    if "gradient_boosting" in selected_models:
        available_models["gradient_boosting"] = {
            "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "use_scaling": False
        }
    
    # AdaBoost
    if "adaboost" in selected_models:
        available_models["adaboost"] = {
            "model": AdaBoostClassifier(n_estimators=100, random_state=42),
            "use_scaling": False
        }
    
    # Extra Trees
    if "extra_trees" in selected_models:
        available_models["extra_trees"] = {
            "model": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weight_param),
            "use_scaling": False
        }
    
    # XGBoost (if available)
    if "xgboost" in selected_models:
        if XGBOOST_AVAILABLE:
            n_classes = len(np.unique(y_train))
            if n_classes > 2:
                available_models["xgboost"] = {
                    "model": XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
                    "use_scaling": False
                }
            else:
                available_models["xgboost"] = {
                    "model": XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss'),
                    "use_scaling": False
                }
        else:
            logging.warning("XGBoost requested but not installed. Skipping.")
    
    # LightGBM (if available)
    if "lightgbm" in selected_models:
        if LIGHTGBM_AVAILABLE:
            available_models["lightgbm"] = {
                "model": LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
                "use_scaling": False
            }
        else:
            logging.warning("LightGBM requested but not installed. Skipping.")
    
    # Check if we have any models to train
    if not available_models:
        return jsonify({"error": "No valid models selected. Please select at least one model."}), 400
    
    logging.info(f"Training {len(available_models)} models: {list(available_models.keys())}")

    accuracies = {"validation": {}, "test": {}}
    confusion_matrices = {"validation": {}, "test": {}}  # Store confusion matrices
    f1_scores = {"validation": {}, "test": {}}  # Store F1 scores
    auc_scores = {"validation": {}, "test": {}}  # Store AUC scores

    def train_and_eval(model, use_scaling, name):
        try:
            logging.info(f"[PROGRESS] Training {name} model...")
            # Use parallel processing where available
            if hasattr(model, 'n_jobs') and name not in ('svm', 'logistic_regression'):
               model.n_jobs = -1
               logging.info(f"Using parallel processing for {name} with n_jobs=-1")  
            if use_scaling:
                model.fit(X_train_scaled, y_train)
                val_pred = model.predict(X_val_scaled)
                test_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
            accuracies["validation"][name] = accuracy_score(y_val, val_pred)
            accuracies["test"][name] = accuracy_score(y_test, test_pred)
            # Calculate confusion matrices
            confusion_matrices["validation"][name] = confusion_matrix(y_val, val_pred).tolist()
            confusion_matrices["test"][name] = confusion_matrix(y_test, test_pred).tolist()
            # Calculate F1 scores (weighted average for multi-class)
            f1_scores["validation"][name] = f1_score(y_val, val_pred, average='weighted')
            f1_scores["test"][name] = f1_score(y_test, test_pred, average='weighted')
            # Calculate AUC scores
            try:
                # Get the classes that the model was trained on
                train_classes = np.unique(y_train)
                n_classes = len(train_classes)
                
                # Get probability predictions for AUC
                if hasattr(model, 'predict_proba'):
                    if use_scaling:
                        val_proba = model.predict_proba(X_val_scaled)
                        test_proba = model.predict_proba(X_test_scaled)
                    else:
                        val_proba = model.predict_proba(X_val)
                        test_proba = model.predict_proba(X_test)
                    
                    # Multi-class AUC with OVR strategy
                    if n_classes > 2:
                        # Use labels parameter to specify the classes
                        auc_scores["validation"][name] = roc_auc_score(y_val, val_proba, multi_class='ovr', average='weighted', labels=train_classes)
                        auc_scores["test"][name] = roc_auc_score(y_test, test_proba, multi_class='ovr', average='weighted', labels=train_classes)
                    else:
                        auc_scores["validation"][name] = roc_auc_score(y_val, val_proba[:, 1])
                        auc_scores["test"][name] = roc_auc_score(y_test, test_proba[:, 1])
                elif hasattr(model, 'decision_function'):
                    # For models like SVM that have decision_function
                    if use_scaling:
                        val_decision = model.decision_function(X_val_scaled)
                        test_decision = model.decision_function(X_test_scaled)
                    else:
                        val_decision = model.decision_function(X_val)
                        test_decision = model.decision_function(X_test)
                    
                    if n_classes > 2:
                        auc_scores["validation"][name] = roc_auc_score(y_val, val_decision, multi_class='ovr', average='weighted', labels=train_classes)
                        auc_scores["test"][name] = roc_auc_score(y_test, test_decision, multi_class='ovr', average='weighted', labels=train_classes)
                    else:
                        auc_scores["validation"][name] = roc_auc_score(y_val, val_decision)
                        auc_scores["test"][name] = roc_auc_score(y_test, test_decision)
                else:
                    auc_scores["validation"][name] = None
                    auc_scores["test"][name] = None
            except Exception as auc_ex:
                logging.warning(f"Could not calculate AUC for {name}: {str(auc_ex)}")
                auc_scores["validation"][name] = None
                auc_scores["test"][name] = None
            logging.info(f"[PROGRESS] {name} model training completed")
        except Exception as ex:
            logging.exception(f"Error training model {name}")

    log_progress("Starting model training in parallel threads")
    
    threads = []
    for model_name, model_config_item in available_models.items():
        t = threading.Thread(
            target=train_and_eval, 
            args=(model_config_item["model"], model_config_item["use_scaling"], model_name)
        )
        threads.append(t)
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    log_progress("All models trained successfully")

    logging.info(f"Training accuracies: {accuracies}")

    # Collect all trained models
    models_trained = {name: config["model"] for name, config in available_models.items()}

    log_progress("Saving model configuration")
    
    model_config.update({
        "features": features,
        "models": models_trained,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "accuracies": accuracies,
        "confusion_matrices": confusion_matrices,
        "f1_scores": f1_scores,
        "auc_scores": auc_scores,
        "target_type": target_type,
        "target_enc": target_enc,
        "scaler": scaler,
        "label_encoders": le_dict,
        "balancing_applied": balancing_applied,
        "balancing_method": balancing_method_used,
        "smote_applied": balancing_applied  # Legacy support
    })

    features_info = []
    for col in features:
        if col in numeric_features:
            features_info.append({"name": col, "type": "numeric"})
        else:
            classes = model_config["label_encoders"][col].classes_.tolist()
            features_info.append({"name": col, "type": "categorical", "classes": classes})

    log_progress("Training process completed successfully", 2)  # Final step counts as 2 to reach 100%
    
    # Get class labels for confusion matrix display
    class_labels = None
    if target_enc is not None:
        class_labels = target_enc.classes_.tolist()
    
    # Build message based on balancing method
    balancing_messages = {
        "none": "",
        "class_weight": " with Class Weights applied",
        "smote": " with SMOTE applied",
        "smote_enn": " with SMOTE-ENN applied",
        "adasyn": " with ADASYN applied",
        "borderline_smote": " with Borderline-SMOTE applied",
        "random_undersample": " with Random Undersampling applied",
        "tomek": " with Tomek Links applied",
        "smote_tomek": " with SMOTE+Tomek applied"
    }
    balancing_msg = balancing_messages.get(balancing_method_used, "")
    
    return jsonify({
        "message": "Models trained" + balancing_msg,
        "accuracies": accuracies,
        "f1_scores": f1_scores,
        "auc_scores": auc_scores,
        "confusion_matrices": confusion_matrices,
        "class_labels": class_labels,
        "features_info": features_info,
        "target_type": target_type,
        "balancing_applied": balancing_applied,
        "balancing_method": balancing_method_used,
        "smote_applied": balancing_applied,  # Legacy support
        "split_info": {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "split_ratio": f"{train_size}% / {val_test_each}% / {val_test_each}%"
        }
    })

@app.route("/get_confusion_matrix", methods=["POST"])
def get_confusion_matrix():
    """Get confusion matrix for a specific model"""
    req = request.get_json()
    model_name = req.get("model_name")
    dataset_type = req.get("dataset_type", "test")  # "validation" or "test"
    
    if not model_config.get("confusion_matrices"):
        return jsonify({"error": "No models trained yet."}), 400
    
    if model_name and model_name not in model_config["confusion_matrices"].get(dataset_type, {}):
        return jsonify({"error": f"Model '{model_name}' not found."}), 400
    
    # Get class labels
    class_labels = None
    if model_config.get("target_enc") is not None:
        class_labels = model_config["target_enc"].classes_.tolist()
    
    if model_name:
        # Return single model's confusion matrix
        return jsonify({
            "confusion_matrix": model_config["confusion_matrices"][dataset_type][model_name],
            "class_labels": class_labels,
            "model_name": model_name,
            "dataset_type": dataset_type
        })
    else:
        # Return all models' confusion matrices
        return jsonify({
            "confusion_matrices": model_config["confusion_matrices"][dataset_type],
            "class_labels": class_labels,
            "dataset_type": dataset_type
        })

@app.route("/export_model", methods=["POST"])
def export_model():
    req = request.get_json()
    base_name = req.get("model_name", "untitled")
    try:
        exported_files = []
        logging.info(f"Exporting models with accuracies: {model_config.get('accuracies')}")
        for algo, model in model_config["models"].items():
            filename = f"{base_name}_{algo}.pkl"
            filepath = os.path.join(models_dir, filename)
            algo_accuracies = {
                "validation": model_config["accuracies"]["validation"].get(algo),
                "test": model_config["accuracies"]["test"].get(algo)
            }
            model_data = {
                "model": model,
                "scaler": model_config["scaler"],
                "label_encoders": model_config["label_encoders"],
                "target_enc": model_config["target_enc"],
                "features": model_config["features"],
                "numeric_features": model_config["numeric_features"],
                "accuracies": algo_accuracies
            }
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
            exported_files.append(filename)
        return jsonify({
            "message": f"Models exported to {models_dir}",
            "files": exported_files,
            "path": models_dir
        })
    except Exception as e:
        logging.exception("Export failed")
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if not model_config:
        return jsonify({"error": "Models not trained yet. Please run /configure first."}), 400
    req = request.get_json()
    values = req.get("values")
    algo = req.get("algorithm")
    if not values:
        return jsonify({"error": "No input values provided."}), 400
    if algo not in model_config["models"]:
        available_models = list(model_config["models"].keys())
        return jsonify({"error": f"Invalid algorithm selection. Available models: {', '.join(available_models)}"}), 400
    feature_names = model_config["features"]
    try:
        input_data = {feat: [values.get(feat)] for feat in feature_names}
        input_df = pd.DataFrame(input_data, columns=feature_names)
    except Exception as e:
        return jsonify({"error": f"Error building input data: {e}"}), 400

    for col in model_config["categorical_features"]:
        if col in input_df.columns:
            le = model_config["label_encoders"].get(col)
            if le:
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except Exception as e:
                    return jsonify({"error": f"Error encoding feature '{col}': {e}"}), 400

    # Models that require scaling
    models_needing_scaling = ["logistic_regression", "svm", "knn", "naive_bayes"]
    
    if algo in models_needing_scaling:
        try:
            input_transformed = model_config["scaler"].transform(input_df)
        except Exception as e:
            return jsonify({"error": f"Error scaling input data: {e}"}), 400
    else:
        input_transformed = input_df

    model = model_config["models"][algo]
    try:
        pred = model.predict(input_transformed)
        if model_config["target_type"] == "categorical":
            attack_prediction = model_config["target_enc"].inverse_transform(pred)[0]
        else:
            attack_prediction = pred[0]
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500
    return jsonify({"prediction": attack_prediction})

@app.route("/predict_loaded/", methods=["POST"])
def predict_loaded_endpoint():
    logging.info("Received predict_loaded request")
    try:
        req = request.get_json()
        values = req.get("values")
        if not values:
            logging.error("No input values provided")
            return jsonify({"error": "No input values provided."}), 400

        model_features = loaded_model_config.get("features", [])
        numeric_features = loaded_model_config.get("numeric_features", [])
        categorical_features = [f for f in model_features if f not in numeric_features]

        input_data = {}
        for feat in model_features:
            val = values.get(feat)
            if isinstance(val, str):
                val = val.strip()
            if feat in numeric_features:
                try:
                    val = float(val)
                except Exception:
                    return jsonify({"error": f"Invalid numeric value for feature '{feat}'."}), 400
            input_data[feat] = [val]

        input_df = pd.DataFrame(input_data, columns=model_features)
        for feat in numeric_features:
            input_df[feat] = input_df[feat].astype(float)

        if "label_encoders" in loaded_model_config:
            for col in categorical_features:
                if col in input_df.columns:
                    encoder = loaded_model_config["label_encoders"].get(col)
                    if encoder:
                        val_str = str(input_df[col].iloc[0])
                        if val_str not in encoder.classes_:
                            error_msg = (f"Value '{val_str}' for feature '{col}' not found in encoder classes. "
                                         f"Expected one of: {encoder.classes_.tolist()}")
                            logging.error(error_msg)
                            return jsonify({"error": error_msg}), 400
                        else:
                            input_df[col] = encoder.transform(input_df[col].astype(str))

        model = loaded_model_config["model"]

        if hasattr(model, "feature_names_in_"):
            input_transformed = pd.DataFrame(input_df, columns=model.feature_names_in_)
        else:
            if isinstance(model, (LogisticRegression, LinearSVC, KNeighborsClassifier)):
                if "scaler" in loaded_model_config and loaded_model_config["scaler"] is not None:
                    input_transformed = loaded_model_config["scaler"].transform(input_df)
                else:
                    input_transformed = input_df
            else:
                input_transformed = input_df

        pred = model.predict(input_transformed)
        if loaded_model_config.get("target_enc"):
            prediction = loaded_model_config["target_enc"].inverse_transform(pred)[0]
        else:
            prediction = pred[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": f"Prediction error: {e}"}), 500

@app.route("/load_model", methods=["POST"])
def load_model_endpoint():
    if "model_file" not in request.files:
        return jsonify({"error": "Model file is required."}), 400

    model_file = request.files["model_file"]
    try:
        model_data = pickle.load(model_file)
    except Exception as e:
        return jsonify({"error": f"Error loading model file: {str(e)}"}), 500

    # Validate that the model file contains necessary information
    required_keys = ["features", "model", "target_enc"]
    missing_keys = [key for key in required_keys if key not in model_data]
    if missing_keys:
        return jsonify({"error": f"Invalid model file. Missing required data: {', '.join(missing_keys)}"}), 400

    global loaded_model_config
    loaded_model_config = model_data

    # Process accuracies for display
    accuracies = model_data.get("accuracies", {})
    if not isinstance(accuracies, dict):
        accuracies = {"validation": {}, "test": {}}
    else:
        if not isinstance(accuracies.get("validation"), dict):
            val_acc = accuracies.get("validation")
            test_acc = accuracies.get("test")
            accuracies = {"validation": {"accuracy": val_acc}, "test": {"accuracy": test_acc}}

    def format_accuracies(acc_dict):
        formatted = {}
        for algo, acc in acc_dict.items():
            try:
                formatted[algo] = f"{acc * 100:.2f}%"
            except Exception:
                formatted[algo] = "N/A"
        return formatted

    organized_accuracies = {
        "Validation": format_accuracies(accuracies.get("validation", {})),
        "Test": format_accuracies(accuracies.get("test", {}))
    }

    # Create feature info for building the prediction form
    features_info = []
    model_features = model_data.get("features", [])
    
    # Determine feature types based on model data
    for feat in model_features:
        # Check if feature has a label encoder (categorical)
        if "label_encoders" in model_data and feat in model_data["label_encoders"]:
            encoder = model_data["label_encoders"][feat]
            classes = encoder.classes_.tolist()
            features_info.append({"name": feat, "type": "categorical", "classes": classes})
        else:
            # Assume numeric if no label encoder
            features_info.append({"name": feat, "type": "numeric"})
    
    # Track numeric features for client-side conversion
    numeric_features = [feat["name"] for feat in features_info if feat["type"] == "numeric"]
    loaded_model_config["numeric_features"] = numeric_features

    return jsonify({
        "message": "Model loaded successfully.",
        "features_info": features_info,
        "accuracies": organized_accuracies
    })

# ---------------------------
# New Routes for the Home Page and Navigation
# ---------------------------
# New Home Page with Navbar (using home.html)
@app.route("/")
def homepage():
    return render_template("home.html")

@app.route("/dynamic_ids")
def dynamic_ids_page():
    return render_template("dynamic_ids.html")


# ---------------------------
# Main entry point
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
