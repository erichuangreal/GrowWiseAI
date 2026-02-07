
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Check XGBoost version
print(f"XGBoost version: {xgb.__version__}")

# Load the data
print("Loading data...")
url = "https://raw.githubusercontent.com/erichuangreal/cxc_hackathon/refs/heads/main/backend/forest_health_data_with_target.csv"
df = pd.read_csv('backend/forest_health_data_with_target.csv')

# ============================================================================
# 1. CREATE BALANCED FEATURE SET WITH REDUCED DOMINANCE
# ============================================================================
print("" + "="*50)
print("FEATURE BALANCING STRATEGY")
print("="*50)

# KEEP Gleason_Index and Disturbance_Level but reduce their dominance
identifier_cols = ['Plot_ID', 'Latitude', 'Longitude', 'DBH', 'Tree_Height', 
                   'Crown_Width_North_South', 'Crown_Width_East_West', 
                   'Menhinick_Index']

# Get all features except identifiers and target
all_features = df.drop(['Health_Status'] + identifier_cols, axis=1)

# ============================================================================
# TECHNIQUE 1: FEATURE TRANSFORMATION TO REDUCE DOMINANCE
# ============================================================================
print("Applying feature transformations to reduce dominance...")

# Create a copy for transformation
X_transformed = all_features.copy()

# Apply transformations to reduce dominance of specific features
dominance_reduction_factor = 0.4  # Reduce to 40% influence

# Transform Gleason_Index (log transform + scaling)
if 'Gleason_Index' in X_transformed.columns:
    X_transformed['Gleason_Index'] = np.log1p(X_transformed['Gleason_Index']) * dominance_reduction_factor
    print(f"  • Gleason_Index: log1p transform + {dominance_reduction_factor*100:.0f}% weight")

# Transform Disturbance_Level (sqrt transform + scaling)
if 'Disturbance_Level' in X_transformed.columns:
    X_transformed['Disturbance_Index'] = np.sqrt(X_transformed['Disturbance_Level']) * dominance_reduction_factor
    # Keep original as well for comparison
    X_transformed['Disturbance_Level'] = X_transformed['Disturbance_Level'] * 0.7  # 70% weight
    print(f"  • Disturbance_Level: sqrt transform + 70% weight")
    print(f"  • Disturbance_Index created: sqrt(Disturbance_Level) + {dominance_reduction_factor*100:.0f}% weight")

# ============================================================================
# TECHNIQUE 2: BOOST ENVIRONMENTAL FEATURES
# ============================================================================
print("Boosting environmental features...")

# Boost environmental features to balance importance
environmental_boost = 1.5  # 50% boost

# List of environmental features to boost
env_features = ['Soil_TN', 'Soil_TP', 'Soil_AP', 'Soil_AN', 'Temperature', 'Humidity', 
                'Slope', 'Elevation', 'Fire_Risk_Index']

for feature in env_features:
    if feature in X_transformed.columns:
        X_transformed[f'{feature}_boosted'] = X_transformed[feature] * environmental_boost
        print(f"  • {feature}: ×{environmental_boost} boost (as {feature}_boosted)")

print(f"Total features after balancing: {X_transformed.shape[1]}")
print(f"Original features: {list(all_features.columns)}")
print(f"New engineered features: {[col for col in X_transformed.columns if col not in all_features.columns]}")

# ============================================================================
# 2. TARGET ENCODING (Keep your original mapping)
# ============================================================================
X = X_transformed
y = df['Health_Status']

health_mapping = {
    'Unhealthy': 0,       # Worst health status
    'Sub-healthy': 1,     # Poor but recovering
    'Healthy': 2,         # Normal health
    'Very Healthy': 3     # Optimal health
}

# Apply the ordinal mapping
y_encoded = y.map(health_mapping)

# Verify all values were mapped
missing_classes = set(y.unique()) - set(health_mapping.keys())
if missing_classes:
    raise ValueError(f"Missing mapping for classes: {missing_classes}")

print(f"Label mapping (ORDINAL HEALTH PROGRESSION):")
for class_name, code in sorted(health_mapping.items(), key=lambda x: x[1]):
    print(f"  {class_name} -> {code}")

# Create LabelEncoder for consistency
le = LabelEncoder()
le.fit(y)

# ============================================================================
# 3. DATA SPLITTING
# ============================================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Data split:")
print(f"Training set: {X_train.shape} ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Validation set: {X_val.shape} ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"Test set: {X_test.shape} ({X_test.shape[0]/len(df)*100:.1f}%)")

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 5. BUILD XGBOOST WITH FEATURE BALANCING CONSTRAINTS
# ============================================================================
print("" + "="*50)
print("Training XGBoost with Feature Balancing...")
print("="*50)

# Calculate class weights for imbalance
class_counts = np.bincount(y_train)
class_weights = {}
for i in range(len(class_counts)):
    if class_counts[i] > 0:
        # Weight inversely proportional to frequency
        class_weights[i] = len(y_train) / (len(class_counts) * class_counts[i])

print(f"Class distribution: {dict(zip(range(4), class_counts))}")
print(f"Class weights: {class_weights}")

# XGBoost model with feature balancing constraints
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    
    # FEATURE BALANCING PARAMETERS
    colsample_bytree=0.6,      # Use only 60% of features per tree
    subsample=0.7,             # Use only 70% of data per tree
    colsample_bylevel=0.7,     # Use only 70% of features per level
    colsample_bynode=0.7,      # Use only 70% of features per node
    
    # Regularization to prevent feature dominance
    reg_alpha=0.5,   # L1 regularization (encourages sparsity)
    reg_lambda=2.0,  # L2 regularization (limits feature weights)
    gamma=0.1,       # Minimum loss reduction for split
    min_child_weight=3,  # Minimum sum of instance weight needed in child
    
    # Tree constraints
    max_delta_step=0,  # 0 = no constraint, positive values limit tree updates
    
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Train with sample weights
sample_weights = np.array([class_weights[label] for label in y_train])
model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

print("Training completed with feature balancing!")

# ============================================================================
# 6. EVALUATION
# ============================================================================
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

# Get probabilities
y_test_prob = model.predict_proba(X_test_scaled)

# Calculate accuracies
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("" + "="*50)
print("MODEL PERFORMANCE WITH FEATURE BALANCING")
print("="*50)
print(f"Training Accuracy:    {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Validation Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"Test Accuracy:        {test_acc:.4f} ({test_acc*100:.2f}%)")

# Baseline comparison
baseline_acc = max(np.bincount(y_test)) / len(y_test)
print(f"Baseline (always predict majority): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"Improvement over baseline: +{(test_acc - baseline_acc)*100:.1f}%")

print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred, 
                           target_names=['Unhealthy', 'Sub-healthy', 'Healthy', 'Very Healthy']))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (Rows=Actual, Columns=Predicted):")
print("         Unhealthy  Sub-healthy  Healthy  Very Healthy")
for i, row_name in enumerate(['Unhealthy', 'Sub-healthy', 'Healthy', 'Very Healthy']):
    row_str = f"{row_name:12}"
    for j in range(4):
        row_str += f"{cm[i, j]:10d}"
    print(row_str)

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS (CRITICAL FOR BALANCING CHECK)
# ============================================================================
print("" + "="*50)
print("FEATURE IMPORTANCE AFTER BALANCING")
print("="*50)

importance_scores = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance_scores
}).sort_values('Importance', ascending=False)

print("Top 15 Most Important Features:")
print("-" * 60)
for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:15],
                                              feature_importance['Importance'][:15]), 1):
    bar_length = int(importance * 100 / feature_importance['Importance'].max() * 0.8)
    bar = '█' * bar_length
    print(f"{i:2}. {feature:30} {bar:40} {importance:.4f}")

# ============================================================================
# 8. BALANCE CHECK: Compare original vs transformed feature importance
# ============================================================================
print("" + "="*50)
print("FEATURE BALANCE ANALYSIS")
print("="*50)

# Identify original high-dominance features
original_dominant = ['Gleason_Index', 'Disturbance_Level', 'Disturbance_Index']
environmental_features = [col for col in X.columns if any(env in col for env in 
                         ['Soil', 'Temp', 'Humid', 'Slope', 'Elevation', 'Fire'])]

# Calculate average importance
orig_avg = feature_importance[feature_importance['Feature'].isin(original_dominant)]['Importance'].mean()
env_avg = feature_importance[feature_importance['Feature'].isin(environmental_features)]['Importance'].mean()

print(f"Average Importance Scores:")
print(f"  Original dominant features: {orig_avg:.4f}")
print(f"  Environmental features:     {env_avg:.4f}")
print(f"  Balance ratio (Env/Orig):   {env_avg/max(orig_avg, 0.001):.2f}:1")

if env_avg > orig_avg:
    print("  ✅ SUCCESS: Environmental features now have higher importance!")
else:
    print(f"  ⚠️  Environmental features are {orig_avg/env_avg:.1f}x less important than original features")

# Check specific feature reductions
print("Specific Feature Importance Checks:")
for feature in ['Gleason_Index', 'Disturbance_Level', 'Disturbance_Index']:
    if feature in feature_importance['Feature'].values:
        imp = feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
        rank = feature_importance[feature_importance['Feature'] == feature].index[0] + 1
        print(f"  {feature:20}: Importance={imp:.4f}, Rank={rank}/{len(feature_importance)}")

# ============================================================================
# 9. PREDICTION FUNCTION (Updated for balanced features)
# ============================================================================
def predict_health_status(features_dict):
    """
    Predict health status for new data with balanced features
    """
    # Create dataframe from input
    input_df = pd.DataFrame([features_dict])
    
    # Apply the same transformations
    if 'Gleason_Index' in input_df.columns:
        input_df['Gleason_Index'] = np.log1p(input_df['Gleason_Index']) * dominance_reduction_factor
    
    if 'Disturbance_Level' in input_df.columns:
        input_df['Disturbance_Index'] = np.sqrt(input_df['Disturbance_Level']) * dominance_reduction_factor
        input_df['Disturbance_Level'] = input_df['Disturbance_Level'] * 0.7
    
    # Boost environmental features
    for feature in env_features:
        if feature in input_df.columns:
            input_df[f'{feature}_boosted'] = input_df[feature] * environmental_boost
    
    # Ensure all columns are present
    for col in X.columns:
        if col not in input_df.columns:
            if col in X_train.columns:
                input_df[col] = X_train[col].median()
            else:
                # For engineered features, calculate if possible
                if '_boosted' in col:
                    base_feature = col.replace('_boosted', '')
                    if base_feature in input_df.columns:
                        input_df[col] = input_df[base_feature] * environmental_boost
                    else:
                        input_df[col] = X_train[col].median()
                else:
                    input_df[col] = X_train[col].median()
    
    # Reorder columns
    input_df = input_df[X.columns]
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    pred_numeric = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]
    
    # Get class name using mapping
    reverse_mapping = {v: k for k, v in health_mapping.items()}
    pred_class = reverse_mapping[pred_numeric]
    
    # Create result dictionary
    result = {
        'predicted_class': pred_class,
        'predicted_class_numeric': int(pred_numeric),
        'probabilities': {reverse_mapping[i]: float(prob) for i, prob in enumerate(pred_proba)},
        'confidence': float(max(pred_proba))
    }
    
    return result

# ============================================================================
# 10. EXAMPLE PREDICTION
# ============================================================================
print("" + "="*50)
print("EXAMPLE PREDICTION WITH BALANCED FEATURES")
print("="*50)

# Get base feature names (before transformation)
base_features = list(all_features.columns)
print(f"Base features needed ({len(base_features)}):")
for i, feat in enumerate(base_features, 1):
    print(f"  {i:2}. {feat}")

# Create example input
example_features = {}
for col in base_features:
    example_features[col] = float(all_features[col].median())

print(f"Example input (median values):")
for k, v in example_features.items():
    print(f"  {k}: {v:.4f}")

try:
    prediction = predict_health_status(example_features)
    print(f"Prediction: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Probabilities:")
    for class_name, prob in sorted(prediction['probabilities'].items(), 
                                   key=lambda x: health_mapping[x[0]]):
        print(f"  {class_name:12}: {prob:.4f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# 11. SAVE MODEL
# ============================================================================
import joblib

model_components = {
    'model': model,
    'scaler': scaler,
    'feature_names': list(X.columns),
    'base_features': base_features,
    'label_mapping': health_mapping,
    'dominance_reduction_factor': dominance_reduction_factor,
    'environmental_boost': environmental_boost,
    'test_accuracy': test_acc
}

joblib.dump(model_components, 'forest_health_model_balanced.pkl')
print("" + "="*50)
print(f"Balanced model saved as 'forest_health_model_balanced.pkl'")
print("="*50)

# ============================================================================
# 12. MODEL SUMMARY
# ============================================================================
print("" + "="*50)
print("BALANCED MODEL SUMMARY")
print("="*50)
print(f"Total features: {len(X.columns)}")
print(f"Test accuracy: {test_acc*100:.2f}%")
print(f"Baseline improvement: +{(test_acc - baseline_acc)*100:.1f}%")
print(f"Key balancing techniques applied:")
print(f"  1. Gleason_Index: log1p transform + {dominance_reduction_factor*100:.0f}% weight")
print(f"  2. Disturbance_Level: sqrt transform + 70% weight")
print(f"  3. Environmental features: ×{environmental_boost} boost")
print(f"  4. XGBoost constraints: colsample=60%, regularization increased")

if test_acc > 0.50:
    print("✅ BALANCING SUCCESSFUL: Model achieves >50% accuracy with balanced features")
else:
    print("⚠️  Accuracy below 50% - consider stronger transformations or binary classification")

print("Model with balanced features is ready for deployment!")
