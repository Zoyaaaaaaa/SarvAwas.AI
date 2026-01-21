"""
RERA Risk Prediction - Robust Production Training Pipeline
Includes: Data cleaning, advanced preprocessing, hyperparameter tuning, evaluation, and logging
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import re
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ==========================
# LOGGING SETUP
# ==========================
LOG_DIR = Path("training_logs")
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================
# DATA CLEANING UTILITIES
# ==========================

def clean_text(text):
    """Clean project/promoter names"""
    if pd.isna(text):
        return "Unknown"
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'[^\w\s-]', '', text)  # Remove special chars except hyphen
    return text.title()

def extract_text_features(text):
    """Extract features from text fields"""
    if pd.isna(text) or text == "Unknown":
        return {
            'word_count': 0,
            'has_number': 0,
            'length': 0
        }
    
    return {
        'word_count': len(str(text).split()),
        'has_number': int(bool(re.search(r'\d', str(text)))),
        'length': len(str(text))
    }

# ==========================
# ROBUST TRAINER CLASS
# ==========================

class RobustRERATrainer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.df = None
        self.models = {}
        self.label_encoders = {}
        self.feature_cols = []
        self.preprocessors = {}
        self.training_metadata = {
            'timestamp': timestamp,
            'data_shape': None,
            'features_used': None,
            'model_performance': {}
        }
        
    def load_and_merge_data(self):
        """Load all RERA datasets and merge intelligently"""
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING DATASETS")
        logger.info("=" * 80)
        
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for file_path in csv_files:
            try:
                temp_df = pd.read_csv(file_path, low_memory=False)
                temp_df['data_source'] = file_path.stem
                dfs.append(temp_df)
                logger.info(f"✓ Loaded {file_path.name}: {len(temp_df)} rows, {len(temp_df.columns)} cols")
            except Exception as e:
                logger.error(f"✗ Failed to load {file_path.name}: {e}")
        
        self.df = pd.concat(dfs, ignore_index=True)
        logger.info(f"\n📊 Total combined dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    def clean_and_preprocess(self):
        """Comprehensive data cleaning and preprocessing"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA CLEANING & PREPROCESSING")
        logger.info("=" * 80)
        
        df = self.df.copy()
        initial_rows = len(df)
        
        # 1. Remove duplicates
        df.drop_duplicates(inplace=True)
        logger.info(f"✓ Removed {initial_rows - len(df)} duplicate rows")
        
        # 2. Clean text fields
        if 'project_name' in df.columns:
            df['project_name_clean'] = df['project_name'].apply(clean_text)
            name_features = df['project_name'].apply(extract_text_features)
            df['project_name_word_count'] = name_features.apply(lambda x: x['word_count'])
            df['project_name_length'] = name_features.apply(lambda x: x['length'])
            logger.info("✓ Cleaned project names and extracted text features")
        
        if 'promoter_name' in df.columns:
            df['promoter_name_clean'] = df['promoter_name'].apply(clean_text)
            logger.info("✓ Cleaned promoter names")
            
        # 3. Date conversions
        date_cols = ['proposed_date_of_completion', 'revised_proposed_date_of_completion', 
                     'extended_date_of_completion', 'date_last_modified']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        logger.info("✓ Converted date columns")
        
        # 4. Numeric cleaning
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
        logger.info(f"✓ Cleaned {len(numeric_cols)} numeric columns")
        
        # 5. Handle missing values intelligently
        null_summary = df.isnull().sum()
        high_null_cols = null_summary[null_summary > len(df) * 0.8].index.tolist()
        if high_null_cols:
            logger.info(f"⚠ Dropping {len(high_null_cols)} columns with >80% nulls")
            df.drop(columns=high_null_cols, inplace=True)
            
        self.df = df
        logger.info(f"\n✓ Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        
    def engineer_features(self):
        """Advanced feature engineering"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        df = self.df.copy()
        
        # === TIMELINE FEATURES ===
        if 'proposed_date_of_completion' in df.columns and 'revised_proposed_date_of_completion' in df.columns:
            df['delay_days'] = (df['revised_proposed_date_of_completion'] - 
                               df['proposed_date_of_completion']).dt.days.fillna(0).clip(lower=0, upper=7300)
            df['has_delay'] = (df['delay_days'] > 0).astype(int)
            df['delay_severity'] = pd.cut(df['delay_days'], 
                                         bins=[-1, 0, 180, 365, 730, 10000],
                                         labels=[0, 1, 2, 3, 4])
            df['delay_severity'] = df['delay_severity'].cat.add_categories([-1]).fillna(-1).astype(int).replace(-1, 0)
            logger.info("✓ Created timeline delay features")
            
        if 'extended_date_of_completion' in df.columns and 'proposed_date_of_completion' in df.columns:
            df['extension_days'] = (df['extended_date_of_completion'] - 
                                   df['proposed_date_of_completion']).dt.days.fillna(0).clip(lower=0)
            df['has_extension'] = (df['extension_days'] > 0).astype(int)
            logger.info("✓ Created extension features")
            
        # === MARKET DEMAND FEATURES ===
        if 'number_of_appartments' in df.columns and 'number_of_booked_appartments' in df.columns:
            df['booking_ratio'] = (df['number_of_booked_appartments'] / 
                                  df['number_of_appartments'].replace(0, 1)).clip(0, 1).fillna(0)
            df['inventory_left'] = (df['number_of_appartments'] - 
                                   df['number_of_booked_appartments']).clip(lower=0).fillna(0)
            df['market_absorption_risk'] = (df['booking_ratio'] < 0.2).astype(int)
            df['high_demand_signal'] = (df['booking_ratio'] > 0.8).astype(int)
            logger.info("✓ Created market demand features")
            
        # === STRUCTURAL COMPLEXITY ===
        if 'sanctioned_fsi' in df.columns:
            df['fsi_intensity'] = pd.to_numeric(df['sanctioned_fsi'], errors='coerce').fillna(0)
            df['fsi_category'] = pd.cut(df['fsi_intensity'], 
                                       bins=[-1, 1, 2, 3, 100],
                                       labels=[0, 1, 2, 3])
            df['fsi_category'] = df['fsi_category'].cat.add_categories([-1]).fillna(-1).astype(int).replace(-1, 0)
            logger.info("✓ Created FSI features")
            
        if 'number_of_sanctioned_floors' in df.columns:
            df['vertical_complexity'] = pd.to_numeric(df['number_of_sanctioned_floors'], 
                                                     errors='coerce').fillna(1)
            df['is_high_rise'] = (df['vertical_complexity'] > 10).astype(int)
            logger.info("✓ Created vertical complexity features")
            
        # === LEGAL & COMPLIANCE ===
        if 'cases_count' in df.columns:
            df['legal_risk_score'] = pd.to_numeric(df['cases_count'], errors='coerce').fillna(0)
            df['has_litigation'] = (df['legal_risk_score'] > 0).astype(int)
            logger.info("✓ Created legal risk features")
            
        # === LOCATION ENCODING ===
        if 'district' in df.columns:
            # District-level aggregated features - only use columns that exist
            agg_dict = {}
            if 'delay_days' in df.columns:
                agg_dict['delay_days'] = 'mean'
            if 'booking_ratio' in df.columns:
                agg_dict['booking_ratio'] = 'mean'
            if 'legal_risk_score' in df.columns:
                agg_dict['legal_risk_score'] = 'sum'
                
            if agg_dict:  # Only if we have something toaggregate
                district_stats = df.groupby('district').agg(agg_dict).add_prefix('district_avg_')
                df = df.merge(district_stats, left_on='district', right_index=True, how='left')
                logger.info(f"✓ Created district-level aggregated features: {list(district_stats.columns)}")
            
        # === PROJECT SIZE ===
        if 'number_of_appartments' in df.columns:
            df['project_scale'] = pd.cut(df['number_of_appartments'].fillna(0),
                                        bins=[0, 20, 50, 100, 500, 10000],
                                        labels=[0, 1, 2, 3, 4])
            df['project_scale'] = df['project_scale'].cat.add_categories([-1]).fillna(-1).astype(int).replace(-1, 0)
            logger.info("✓ Created project scale features")
            
        self.df = df
        logger.info(f"\n✓ Feature engineering complete: {len(df.columns)} total columns")
        
    def create_targets(self):
        """Create multi-task prediction targets"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: TARGET VARIABLE CREATION")
        logger.info("=" * 80)
        
        df = self.df.copy()
        
        # Composite Risk Score (0-10)
        risk_score = pd.Series(0.0, index=df.index)
        
        if 'delay_days' in df.columns:
            risk_score += (df['delay_days'] / 365).clip(0, 3)
        if 'booking_ratio' in df.columns:
            risk_score += (1 - df['booking_ratio']) * 3
        if 'legal_risk_score' in df.columns:
            risk_score += (df['legal_risk_score'] / 5).clip(0, 2)
        if 'has_extension' in df.columns:
            risk_score += df['has_extension'] * 2
            
        df['regulatory_risk_score'] = risk_score.clip(0, 10)
        df['completion_probability'] = (1 - (risk_score / 10)).clip(0, 1)
        
        # Delay Risk Classification
        if 'delay_days' in df.columns:
            df['delay_risk_class'] = pd.cut(df['delay_days'], 
                                           bins=[-1, 1, 180, 540, 10000],
                                           labels=[0, 1, 2, 3]).astype(int)
        else:
            df['delay_risk_class'] = 0
            
        # Fill NaNs in targets
        df['completion_probability'] = df['completion_probability'].fillna(0.5)
        df['delay_risk_class'] = df['delay_risk_class'].fillna(0).astype(int)
        
        self.df = df
        
        logger.info(f"✓ Completion Probability - Mean: {df['completion_probability'].mean():.3f}, "
                   f"Std: {df['completion_probability'].std():.3f}")
        logger.info(f"✓ Delay Risk Class distribution:\n{df['delay_risk_class'].value_counts().sort_index()}")
        
    def prepare_features(self):
        """Select and prepare final feature set"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: FEATURE SELECTION")
        logger.info("=" * 80)
        
        # Candidate features
        candidate_features = [
            'booking_ratio', 'inventory_left', 'fsi_intensity', 
            'vertical_complexity', 'legal_risk_score', 'has_delay', 
            'has_extension', 'project_scale', 'delay_severity',
            'market_absorption_risk', 'high_demand_signal',
            'is_high_rise', 'has_litigation', 'fsi_category',
            'project_name_word_count', 'project_name_length',
            'district', 'district_avg_delay_days', 'district_avg_booking_ratio'
        ]
        
        # Filter to existing columns
        self.feature_cols = [c for c in candidate_features if c in self.df.columns]
        
        # Fill NaNs
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
        
        # Encode categoricals
        for col in ['district']:
            if col in self.feature_cols:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                
        logger.info(f"✓ Selected {len(self.feature_cols)} features: {self.feature_cols}")
        
    def train_models(self):
        """Train multiple models with cross-validation"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: MODEL TRAINING")
        logger.info("=" * 80)
        
        X = self.df[self.feature_cols]
        
        # === TASK 1: Completion Probability (Regression) ===
        logger.info("\n🎯 Training Completion Probability Model (Regression)")
        y_prob = self.df['completion_probability']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_prob, test_size=0.2, random_state=42
        )
        
        model_prob = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            importance_type='gain',
            random_state=42,
            verbose=-1
        )
        
        model_prob.fit(X_train, y_train)
        self.models['completion_probability'] = model_prob
        
        # Evaluation
        y_pred = model_prob.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"  ✓ MAE: {mae:.4f}")
        logger.info(f"  ✓ RMSE: {rmse:.4f}")
        logger.info(f"  ✓ R²: {r2:.4f}")
        
        self.training_metadata['model_performance']['completion_probability'] = {
            'mae': mae, 'rmse': rmse, 'r2': r2
        }
        
        # === TASK 2: Delay Risk Classification ===
        logger.info("\n🎯 Training Delay Risk Classifier")
        y_delay = self.df['delay_risk_class']
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y_delay, test_size=0.2, random_state=42, stratify=y_delay
        )
        
        model_delay = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        
        model_delay.fit(X_train2, y_train2)
        self.models['delay_risk_class'] = model_delay
        
        # Evaluation
        y_pred2 = model_delay.predict(X_test2)
        y_proba2 = model_delay.predict_proba(X_test2)
        
        acc = accuracy_score(y_test2, y_pred2)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test2, y_pred2, average='weighted')
        
        logger.info(f"  ✓ Accuracy: {acc:.4f}")
        logger.info(f"  ✓ Weighted F1: {f1:.4f}")
        logger.info(f"  ✓ Precision: {precision:.4f}")
        logger.info(f"  ✓ Recall: {recall:.4f}")
        
        logger.info("\n  Classification Report:")
        logger.info("\n" + classification_report(y_test2, y_pred2, 
                                                 target_names=['Minimal', 'Short', 'Moderate', 'Severe']))
        
        self.training_metadata['model_performance']['delay_risk_class'] = {
            'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall
        }
        
        # Feature Importance
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model_prob.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\n📊 Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
            
        self.training_metadata['feature_importance'] = importance_df.to_dict('records')
        
    def test_scenarios(self):
        """Test model with 3 realistic scenarios"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: SCENARIO TESTING")
        logger.info("=" * 80)
        
        scenarios = [
            {
                "name": "Low Risk Project (High Booking, No Delays)",
                "data": {
                    'booking_ratio': 0.95,
                    'inventory_left': 5,
                    'fsi_intensity': 2.5,
                    'vertical_complexity': 12,
                    'legal_risk_score': 0,
                    'has_delay': 0,
                    'has_extension': 0,
                    'project_scale': 2,
                    'delay_severity': 0,
                    'market_absorption_risk': 0,
                    'high_demand_signal': 1,
                    'is_high_rise': 1,
                    'has_litigation': 0,
                    'fsi_category': 2,
                    'project_name_word_count': 3,
                    'project_name_length': 25,
                    'district': 0,
                    'district_avg_delay_days': 50,
                    'district_avg_booking_ratio': 0.75
                }
            },
            {
                "name": "Moderate Risk Project (Medium Booking, Some Delays)",
                "data": {
                    'booking_ratio': 0.55,
                    'inventory_left': 45,
                    'fsi_intensity': 2.0,
                    'vertical_complexity': 8,
                    'legal_risk_score': 1,
                    'has_delay': 1,
                    'has_extension': 0,
                    'project_scale': 2,
                    'delay_severity': 2,
                    'market_absorption_risk': 0,
                    'high_demand_signal': 0,
                    'is_high_rise': 0,
                    'has_litigation': 1,
                    'fsi_category': 1,
                    'project_name_word_count': 4,
                    'project_name_length': 30,
                    'district': 0,
                    'district_avg_delay_days': 200,
                    'district_avg_booking_ratio': 0.60
                }
            },
            {
                "name": "High Risk Project (Low Booking, Multiple Issues)",
                "data": {
                    'booking_ratio': 0.15,
                    'inventory_left': 170,
                    'fsi_intensity': 1.5,
                    'vertical_complexity': 15,
                    'legal_risk_score': 5,
                    'has_delay': 1,
                    'has_extension': 1,
                    'project_scale': 3,
                    'delay_severity': 3,
                    'market_absorption_risk': 1,
                    'high_demand_signal': 0,
                    'is_high_rise': 1,
                    'has_litigation': 1,
                    'fsi_category': 1,
                    'project_name_word_count': 2,
                    'project_name_length': 15,
                    'district': 1,
                    'district_avg_delay_days': 400,
                    'district_avg_booking_ratio': 0.40
                }
            }
        ]
        
        for scenario in scenarios:
            logger.info(f"\n🧪 Testing: {scenario['name']}")
            
            # Prepare input
            input_df = pd.DataFrame([scenario['data']])[self.feature_cols]
            
            # Predict
            comp_prob = self.models['completion_probability'].predict(input_df)[0]
            delay_class = self.models['delay_risk_class'].predict(input_df)[0]
            delay_proba = self.models['delay_risk_class'].predict_proba(input_df)[0]
            
            risk_level = "LOW" if comp_prob > 0.7 else "MODERATE" if comp_prob > 0.4 else "HIGH"
            delay_labels = ["Minimal", "Short", "Moderate", "Severe"]
            
            logger.info(f"  ✓ Completion Probability: {comp_prob:.2%}")
            logger.info(f"  ✓ Risk Level: {risk_level}")
            logger.info(f"  ✓ Delay Risk: {delay_labels[delay_class]}")
            logger.info(f"  ✓ Delay Probabilities: {dict(zip(delay_labels, delay_proba))}")
            
    def save_artifacts(self):
        """Save all training artifacts"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: SAVING ARTIFACTS")
        logger.info("=" * 80)
        
        output_dir = Path("app/models")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save models
        with open(output_dir / "models.pkl", "wb") as f:
            pickle.dump(self.models, f)
        logger.info("✓ Saved models.pkl")

        # Generate and Save SHAP Explainer (Task 1: Completion Probability)
        import shap
        logger.info("Generating SHAP explainer...")
        # Use a background dataset for the explainer (optional but good for robustness)
        # For TreeExplainer with LightGBM, we can just pass the model
        explainer = shap.TreeExplainer(self.models['completion_probability'])
        
        with open(output_dir / "shap_explainer.pkl", "wb") as f:
            pickle.dump(explainer, f)
        logger.info("✓ Saved shap_explainer.pkl")
        
        # Save metadata
        metadata = {
            'feature_cols': self.feature_cols,
            'label_encoders': self.label_encoders,
            'target_types': {
                'completion_probability': 'regression',
                'delay_risk_class': 'classification'
            },
            'training_metadata': self.training_metadata
        }
        
        with open(output_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        logger.info("✓ Saved metadata.pkl")
        
        # Save metrics report
        metrics_file = Path("model_training_results.txt")
        with open(metrics_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("RERA RISK PREDICTION MODEL - TRAINING RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Training Timestamp: {timestamp}\n")
            f.write(f"Total Features: {len(self.feature_cols)}\n")
            f.write(f"Dataset Size: {len(self.df)} projects\n\n")
            
            f.write("COMPLETION PROBABILITY MODEL (Regression)\n")
            f.write("-" * 80 + "\n")
            perf = self.training_metadata['model_performance']['completion_probability']
            f.write(f"MAE: {perf['mae']:.4f}\n")
            f.write(f"RMSE: {perf['rmse']:.4f}\n")
            f.write(f"R²: {perf['r2']:.4f}\n\n")
            
            f.write("DELAY RISK CLASSIFIER\n")
            f.write("-" * 80 + "\n")
            perf = self.training_metadata['model_performance']['delay_risk_class']
            f.write(f"Accuracy: {perf['accuracy']:.4f}\n")
            f.write(f"Weighted F1: {perf['f1']:.4f}\n")
            f.write(f"Precision: {perf['precision']:.4f}\n")
            f.write(f"Recall: {perf['recall']:.4f}\n\n")
            
            f.write("TOP 10 FEATURES BY IMPORTANCE\n")
            f.write("-" * 80 + "\n")
            for feat in self.training_metadata['feature_importance'][:10]:
                f.write(f"{feat['feature']}: {feat['importance']:.2f}\n")
                
        logger.info(f"✓ Saved {metrics_file}")
        logger.info(f"\n✅ All artifacts saved to {output_dir.absolute()}")
        
    def run_full_pipeline(self):
        """Execute complete training pipeline"""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 RERA RISK PREDICTION - ROBUST TRAINING PIPELINE")
        logger.info("=" * 80)
        
        try:
            self.load_and_merge_data()
            self.clean_and_preprocess()
            self.engineer_features()
            self.create_targets()
            self.prepare_features()
            self.train_models()
            self.test_scenarios()
            self.save_artifacts()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"📝 Training log saved to: {log_file}")
            
        except Exception as e:
            logger.error(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

# ==========================
# MAIN EXECUTION
# ==========================

if __name__ == "__main__":
    # Use data folder within models_rera directory
    data_dir = Path(__file__).parent / "data"
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error(f"Looked in: {data_dir.absolute()}")
        exit(1)
        
    trainer = RobustRERATrainer(data_dir)
    trainer.run_full_pipeline()
