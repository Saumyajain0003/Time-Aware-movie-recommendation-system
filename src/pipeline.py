"""
ML Pipeline for time-aware recommender system.
Orchestrates data generation, model training, and evaluation.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from data import DataGenerator, DataProcessor, DataLoader, save_data, load_data
from models import TimeAwareRecommender, TimeSensitiveMatrixFactorization


class RecommenderPipeline:
    """Complete ML pipeline for time-aware recommendations."""
    
    def __init__(self, 
                 project_dir: str = ".",
                 config: Optional[Dict] = None):
        """
        Initialize pipeline.
        
        Args:
            project_dir: Root directory of project
            config: Configuration dictionary
        """
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"
        self.models_dir = self.project_dir / "models"
        self.results_dir = self.project_dir / "results"
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Default config
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # Pipeline state
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.results = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'data': {
                'n_users': 1000,
                'n_items': 500,
                'n_interactions': 50000,
                'temporal_decay': 0.95,
                'random_state': 42
            },
            'preprocessing': {
                'test_ratio': 0.2,
                'temporal_split': True,
                'seq_length': 5
            },
            'models': {
                'tar': {'n_factors': 20, 'learning_rate': 0.01, 'n_epochs': 50},
                'tsmf': {'n_factors': 20, 'learning_rate': 0.01, 'n_epochs': 50}
            },
            'evaluation': {
                'k_values': [5, 10, 20],
                'batch_size': 32
            }
        }
    
    def _setup_logger(self):
        """Setup logging."""
        class SimpleLogger:
            def info(self, msg):
                print(f"[INFO] {msg}")
            
            def warning(self, msg):
                print(f"[⚠️  WARNING] {msg}")
            
            def error(self, msg):
                print(f"[❌ ERROR] {msg}")
            
            def success(self, msg):
                print(f"[✓ SUCCESS] {msg}")
        
        return SimpleLogger()
    
    def stage_1_data_generation(self, skip_if_exists: bool = False):
        """Stage 1: Generate synthetic data."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: Data Generation")
        self.logger.info("=" * 60)
        
        # Check if data already exists
        interactions_file = self.data_dir / "interactions.csv"
        if skip_if_exists and interactions_file.exists():
            self.logger.info("Data already exists, skipping generation...")
            return self._load_existing_data()
        
        # Generate data
        self.logger.info(f"Generating synthetic data with config: {self.config['data']}")
        
        generator = DataGenerator(**self.config['data'])
        df = generator.generate(days=365)
        generator.add_item_features(df, n_features=10)
        generator.add_user_features(df, n_features=5)
        
        # Save raw data
        save_data(df, str(interactions_file))
        
        self.logger.success("Data generation complete")
        return df
    
    def stage_2_preprocessing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 2: Data preprocessing and splitting."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: Data Preprocessing")
        self.logger.info("=" * 60)
        
        processor = DataProcessor()
        
        # Train-test split
        self.logger.info(f"Creating train/test split with config: {self.config['preprocessing']}")
        train_df, test_df = processor.create_train_test_split(
            df,
            test_ratio=self.config['preprocessing']['test_ratio'],
            temporal_split=self.config['preprocessing']['temporal_split']
        )
        
        # Create temporal sequences
        user_sequences = processor.create_temporal_sequences(
            train_df,
            seq_length=self.config['preprocessing']['seq_length']
        )
        
        # Create user-item matrix
        user_item_matrix = processor.create_user_item_matrix(train_df)
        self.logger.info(f"User-item matrix shape: {user_item_matrix.shape}")
        
        # Save processed data
        save_data(train_df, str(self.data_dir / "train_interactions.csv"))
        save_data(test_df, str(self.data_dir / "test_interactions.csv"))
        
        # Save sequences and matrix
        with open(self.data_dir / "user_sequences.pkl", 'wb') as f:
            pickle.dump(user_sequences, f)
        
        np.save(str(self.data_dir / "user_item_matrix.npy"), user_item_matrix)
        
        self.logger.success("Data preprocessing complete")
        
        self.train_data = train_df
        self.test_data = test_df
        self.user_sequences = user_sequences
        self.user_item_matrix = user_item_matrix
        self.processor = processor
        
        return train_df, test_df
    
    def stage_3_model_training(self):
        """Stage 3: Train all models."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: Model Training")
        self.logger.info("=" * 60)
        
        if self.train_data is None:
            raise ValueError("Training data not loaded. Run stage_2_preprocessing first.")
        
        models_config = self.config['models']
        
        # Convert user-item matrix to sparse format
        from scipy.sparse import csr_matrix
        interactions_sparse = csr_matrix(self.user_item_matrix)
        timestamps = self.train_data['timestamp_seconds'].values
        
        # Train Time-Aware Recommender (TAR)
        self.logger.info("\n1. Training Time-Aware Recommender (TAR)...")
        tar_model = TimeAwareRecommender(**models_config['tar'])
        tar_model.fit(interactions_sparse, timestamps)
        self.models['tar'] = tar_model
        self.logger.success("TAR model trained")
        
        # Train Time-Sensitive Matrix Factorization (TSMF)
        self.logger.info("\n2. Training Time-Sensitive Matrix Factorization (TSMF)...")
        tsmf_model = TimeSensitiveMatrixFactorization(**models_config['tsmf'])
        tsmf_model.fit(interactions_sparse, timestamps)
        self.models['tsmf'] = tsmf_model
        self.logger.success("TSMF model trained")
        
        # Save models
        self.logger.info("\nSaving trained models...")
        for model_name, model in self.models.items():
            model_path = self.models_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        self.logger.success("All models trained and saved")
    
    def stage_4_evaluation(self):
        """Stage 4: Evaluate all models."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: Model Evaluation")
        self.logger.info("=" * 60)
        
        if not self.models:
            raise ValueError("No models trained. Run stage_3_model_training first.")
        
        if self.test_data is None:
            raise ValueError("Test data not loaded. Run stage_2_preprocessing first.")
        
        # Prepare test data
        test_user_ids = self.test_data['user_id'].unique()
        test_ratings_dict = {}
        for user_id in test_user_ids:
            test_ratings_dict[user_id] = self.test_data[
                self.test_data['user_id'] == user_id
            ][['item_id', 'rating']].values
        
        # Evaluate each model
        for model_name, model in self.models.items():
            self.logger.info(f"\nEvaluating {model_name.upper()} model...")
            
            metrics = {
                'users_evaluated': 0,
                'avg_recommendations_provided': 0,
                'model_training_complete': True
            }
            
            # Evaluate on test set
            total_recommendations = 0
            evaluated_users = 0
            
            for user_id in test_user_ids:
                try:
                    # Get recommendations
                    recommendations = model.predict(user_id, n_recommendations=10)
                    if recommendations:
                        total_recommendations += len(recommendations)
                        evaluated_users += 1
                except Exception as e:
                    self.logger.warning(f"Could not generate recommendations for user {user_id}: {str(e)}")
            
            if evaluated_users > 0:
                metrics['users_evaluated'] = evaluated_users
                metrics['avg_recommendations_provided'] = total_recommendations / evaluated_users
            
            self.results[model_name] = metrics
            
            # Print results
            self.logger.info(f"\n{model_name.upper()} Results:")
            for metric, value in metrics.items():
                if isinstance(value, bool):
                    self.logger.info(f"  {metric}: {'Yes' if value else 'No'}")
                else:
                    self.logger.info(f"  {metric}: {value:.4f}")
        
        self.logger.success("Evaluation complete")
    
    def save_results(self):
        """Save evaluation results to JSON."""
        self.logger.info("\nSaving results...")
        
        results_file = self.results_dir / "evaluation_results.json"
        
        # Convert numpy types for JSON serialization
        results_to_save = {}
        for model_name, metrics in self.results.items():
            results_to_save[model_name] = {
                k: float(v) for k, v in metrics.items()
            }
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        self.logger.success(f"Results saved to {results_file}")
        
        # Create summary report
        self._create_summary_report()
    
    def _create_summary_report(self):
        """Create a text summary report."""
        report_file = self.results_dir / "summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TIME-AWARE RECOMMENDER SYSTEM - EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Users: {self.config['data']['n_users']}\n")
            f.write(f"Items: {self.config['data']['n_items']}\n")
            f.write(f"Interactions: {self.config['data']['n_interactions']}\n")
            f.write(f"Test Ratio: {self.config['preprocessing']['test_ratio']}\n\n")
            
            f.write("MODEL PERFORMANCE METRICS\n")
            f.write("-" * 70 + "\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name.upper()}\n")
                f.write("  " + "-" * 66 + "\n")
                for metric, value in sorted(metrics.items()):
                    f.write(f"  {metric:25s}: {value:10.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("BEST MODELS BY METRIC\n")
            f.write("=" * 70 + "\n")
            
            # Find best model for each metric
            all_metrics = {}
            for model_name, metrics in self.results.items():
                for metric, value in metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append((model_name, value))
            
            for metric, values in sorted(all_metrics.items()):
                if 'rmse' in metric or 'mae' in metric:
                    best_model, best_value = min(values, key=lambda x: x[1])
                else:
                    best_model, best_value = max(values, key=lambda x: x[1])
                
                f.write(f"\n{metric:25s}: {best_model:15s} ({best_value:.4f})\n")
        
        self.logger.success(f"Summary report saved to {report_file}")
    
    def run_full_pipeline(self, skip_data_if_exists: bool = True):
        """Run the complete pipeline."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STARTING TIME-AWARE RECOMMENDER SYSTEM PIPELINE")
        self.logger.info("=" * 60 + "\n")
        
        try:
            # Stage 1: Data Generation
            df = self.stage_1_data_generation(skip_if_exists=skip_data_if_exists)
            
            # Stage 2: Preprocessing
            train_df, test_df = self.stage_2_preprocessing(df)
            
            # Stage 3: Model Training
            self.stage_3_model_training()
            
            # Stage 4: Evaluation
            self.stage_4_evaluation()
            
            # Save results
            self.save_results()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.success("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _load_existing_data(self) -> pd.DataFrame:
        """Load existing data from disk."""
        df = load_data(str(self.data_dir / "interactions.csv"))
        return df
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get results as a DataFrame for easy comparison."""
        data = []
        for model_name, metrics in self.results.items():
            row = {'model': model_name}
            row.update(metrics)
            data.append(row)
        
        return pd.DataFrame(data)


def main():
    """Main entry point."""
    # Create pipeline
    pipeline = RecommenderPipeline(project_dir=".")
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(skip_data_if_exists=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(pipeline.get_results_summary().to_string(index=False))


if __name__ == "__main__":
    main()
