"""
Demo script for Time-Aware Recommender System.
Shows how to use the system for generating recommendations and analyzing results.
"""

import sys
import json
from pathlib import Path

from pipeline import RecommenderPipeline
from data import DataGenerator, DataProcessor, load_data
from models import TimeAwareRecommender, TimeSensitiveMatrixFactorization
from metrics import RecommendationMetrics


def demo_1_basic_usage():
    """Demo 1: Basic pipeline execution."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Pipeline Execution")
    print("=" * 70)
    
    # Create and run pipeline
    pipeline = RecommenderPipeline()
    results = pipeline.run_full_pipeline(skip_data_if_exists=True)
    
    print("\n✓ Pipeline executed successfully!")
    print("\nGenerated files:")
    print("  - data/interactions.csv (full dataset)")
    print("  - data/train_interactions.csv (80% training)")
    print("  - data/test_interactions.csv (20% testing)")
    print("  - models/tar_model.pkl (Time-Aware Recommender)")
    print("  - models/tsmf_model.pkl (Time-Sensitive Matrix Factorization)")
    print("  - results/evaluation_results.json (metrics)")
    print("  - results/summary_report.txt (detailed report)")


def demo_2_custom_recommendations():
    """Demo 2: Generate custom recommendations for specific users."""
    print("\n" + "=" * 70)
    print("DEMO 2: Custom Recommendations")
    print("=" * 70)
    
    # Load trained models
    import pickle
    
    with open('models/tar_model.pkl', 'rb') as f:
        tar_model = pickle.load(f)
    
    with open('models/tsmf_model.pkl', 'rb') as f:
        tsmf_model = pickle.load(f)
    
    # Generate recommendations for sample users
    sample_users = [0, 5, 10, 25, 100]
    
    print("\nGenerating recommendations for sample users:")
    print("-" * 70)
    
    for user_id in sample_users:
        print(f"\nUser {user_id}:")
        print(f"\n  TAR Model - Top 5 Recommendations:")
        tar_recs = tar_model.predict(user_id, n_recommendations=5)
        for item_id, score in tar_recs:
            print(f"    Item {item_id:4d} - Score: {score:6.3f}")
        
        print(f"\n  TSMF Model - Top 5 Recommendations:")
        tsmf_recs = tsmf_model.predict(user_id, n_recommendations=5)
        for item_id, score in tsmf_recs:
            print(f"    Item {item_id:4d} - Score: {score:6.3f}")


def demo_3_batch_recommendations():
    """Demo 3: Generate recommendations in batch."""
    print("\n" + "=" * 70)
    print("DEMO 3: Batch Recommendations")
    print("=" * 70)
    
    import pickle
    
    with open('models/tar_model.pkl', 'rb') as f:
        tar_model = pickle.load(f)
    
    # Generate recommendations for multiple users
    user_ids = list(range(0, 20))
    
    print(f"\nGenerating recommendations for {len(user_ids)} users...")
    recommendations = tar_model.predict_batch(user_ids, n_recommendations=10)
    
    print(f"✓ Generated {len(recommendations)} recommendation sets")
    print(f"  Average recommendations per user: {sum(len(recs) for recs in recommendations.values()) / len(recommendations):.1f}")


def demo_4_data_analysis():
    """Demo 4: Analyze the data characteristics."""
    print("\n" + "=" * 70)
    print("DEMO 4: Data Analysis")
    print("=" * 70)
    
    # Load data
    df = load_data('data/interactions.csv')
    train_df = load_data('data/train_interactions.csv')
    test_df = load_data('data/test_interactions.csv')
    
    print("\nDataset Statistics:")
    print("-" * 70)
    print(f"Total interactions: {len(df):,}")
    print(f"  Training: {len(train_df):,} ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Testing: {len(test_df):,} ({100*len(test_df)/len(df):.1f}%)")
    
    print(f"\nUnique users: {df['user_id'].nunique():,}")
    print(f"Unique items: {df['item_id'].nunique():,}")
    
    print(f"\nRating Statistics:")
    print(f"  Mean: {df['rating'].mean():.2f}")
    print(f"  Std:  {df['rating'].std():.2f}")
    print(f"  Min:  {df['rating'].min():.2f}")
    print(f"  Max:  {df['rating'].max():.2f}")
    
    print(f"\nTemporal Statistics:")
    print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"  Days span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    print(f"\nUser Activity:")
    user_counts = df.groupby('user_id').size()
    print(f"  Avg interactions per user: {user_counts.mean():.1f}")
    print(f"  Min interactions: {user_counts.min()}")
    print(f"  Max interactions: {user_counts.max()}")
    
    print(f"\nItem Popularity:")
    item_counts = df.groupby('item_id').size()
    print(f"  Avg interactions per item: {item_counts.mean():.1f}")
    print(f"  Min interactions: {item_counts.min()}")
    print(f"  Max interactions: {item_counts.max()}")


def demo_5_view_results():
    """Demo 5: View evaluation results."""
    print("\n" + "=" * 70)
    print("DEMO 5: Evaluation Results")
    print("=" * 70)
    
    # Load results
    with open('results/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    print("\nModel Performance Summary:")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, bool):
                print(f"  {metric:40s}: {'✓' if value else '✗'}")
            else:
                print(f"  {metric:40s}: {value:10.4f}")
    
    # Load and print summary report
    print("\n" + "=" * 70)
    print("DETAILED REPORT:")
    print("=" * 70)
    
    with open('results/summary_report.txt', 'r') as f:
        print(f.read())


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("TIME-AWARE RECOMMENDER SYSTEM - DEMO")
    print("=" * 70)
    
    demo_choice = input("""
Which demo would you like to run?
  1. Basic pipeline execution
  2. Custom recommendations
  3. Batch recommendations
  4. Data analysis
  5. View results
  6. Run all demos
  
Enter choice (1-6): """).strip()
    
    try:
        if demo_choice == '1':
            demo_1_basic_usage()
        elif demo_choice == '2':
            demo_2_custom_recommendations()
        elif demo_choice == '3':
            demo_3_batch_recommendations()
        elif demo_choice == '4':
            demo_4_data_analysis()
        elif demo_choice == '5':
            demo_5_view_results()
        elif demo_choice == '6':
            demo_1_basic_usage()
            demo_2_custom_recommendations()
            demo_3_batch_recommendations()
            demo_4_data_analysis()
            demo_5_view_results()
        else:
            print("Invalid choice. Running demo 1...")
            demo_1_basic_usage()
    except FileNotFoundError as e:
        print(f"\n⚠️  Error: {e}")
        print("Please run 'python src/pipeline.py' first to generate models and data.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
