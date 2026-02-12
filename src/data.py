"""
Data generation and preprocessing module for time-aware recommender system.
Handles synthetic data generation with temporal patterns and data preprocessing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import pickle
from pathlib import Path


class DataGenerator:
    """Generate synthetic user-item interaction data with temporal patterns."""
    
    def __init__(self, 
                 n_users: int = 1000,
                 n_items: int = 500,
                 n_interactions: int = 50000,
                 temporal_decay: float = 0.95,
                 random_state: int = 42):
        """
        Initialize data generator.
        
        Args:
            n_users: Number of unique users
            n_items: Number of unique items
            n_interactions: Total interactions to generate
            temporal_decay: How much past interactions decay in influence (0-1)
            random_state: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_interactions = n_interactions
        self.temporal_decay = temporal_decay
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate(self, days: int = 365) -> pd.DataFrame:
        """
        Generate synthetic interaction data with temporal patterns.
        
        Args:
            days: Number of days to generate data for
            
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        print(f"Generating {self.n_interactions} interactions for {self.n_users} users and {self.n_items} items...")
        
        # Generate base timestamp
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = []
        
        # Generate interactions
        for _ in range(self.n_interactions):
            user_id = np.random.randint(0, self.n_users)
            item_id = np.random.randint(0, self.n_items)
            
            # Add temporal pattern - some users are more active recently
            days_ago = np.random.exponential(scale=days/3)
            timestamp = end_date - timedelta(days=min(days_ago, days))
            
            # Generate rating (1-5) with some correlation to temporal patterns
            base_rating = np.random.normal(3.5, 1.5)
            # Recent items tend to have slightly higher ratings
            recency_boost = (1.0 - (datetime.now() - timestamp).days / days) * 0.5
            rating = np.clip(base_rating + recency_boost, 1, 5)
            
            data.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Convert timestamp to seconds since start for easier processing
        min_timestamp = df['timestamp'].min()
        df['timestamp_seconds'] = (df['timestamp'] - min_timestamp).dt.total_seconds()
        
        print(f"✓ Generated {len(df)} interactions")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")
        
        return df
    
    def add_item_features(self, df: pd.DataFrame, n_features: int = 10) -> pd.DataFrame:
        """Add synthetic item features to the dataframe."""
        print(f"Adding {n_features} item features...")
        
        # Create item features (simulate item embeddings)
        item_features = np.random.randn(self.n_items, n_features)
        
        # Store features as a dictionary for easy lookup
        self.item_features = {i: item_features[i] for i in range(self.n_items)}
        
        return df
    
    def add_user_features(self, df: pd.DataFrame, n_features: int = 5) -> pd.DataFrame:
        """Add synthetic user features to the dataframe."""
        print(f"Adding {n_features} user features...")
        
        # Create user features (simulate user embeddings)
        user_features = np.random.randn(self.n_users, n_features)
        
        # Store features as a dictionary for easy lookup
        self.user_features = {i: user_features[i] for i in range(self.n_users)}
        
        return df


class DataProcessor:
    """Process and prepare data for model training."""
    
    def __init__(self, temporal_window: int = 604800):
        """
        Initialize data processor.
        
        Args:
            temporal_window: Time window in seconds (default: 7 days = 604800 seconds)
        """
        self.temporal_window = temporal_window
        
    def create_train_test_split(self, 
                               df: pd.DataFrame, 
                               test_ratio: float = 0.2,
                               temporal_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split.
        
        Args:
            df: Input dataframe
            test_ratio: Ratio of test data
            temporal_split: If True, use temporal split (future data as test)
                           If False, use random split
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if temporal_split:
            # Temporal split: use later data as test
            split_point = int(len(df) * (1 - test_ratio))
            train_df = df.iloc[:split_point].copy()
            test_df = df.iloc[split_point:].copy()
            print(f"✓ Temporal split: {len(train_df)} train, {len(test_df)} test")
        else:
            # Random split
            mask = np.random.random(len(df)) < (1 - test_ratio)
            train_df = df[mask].copy()
            test_df = df[~mask].copy()
            print(f"✓ Random split: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df
    
    def create_user_item_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Create user-item rating matrix."""
        n_users = df['user_id'].max() + 1
        n_items = df['item_id'].max() + 1
        
        matrix = np.zeros((n_users, n_items))
        
        for _, row in df.iterrows():
            matrix[row['user_id'], row['item_id']] = row['rating']
        
        return matrix
    
    def create_temporal_sequences(self, 
                                  df: pd.DataFrame,
                                  seq_length: int = 5) -> Dict[int, List]:
        """
        Create temporal sequences of interactions for each user.
        
        Args:
            df: Input dataframe
            seq_length: Length of sequence to create
            
        Returns:
            Dictionary mapping user_id to list of item sequences
        """
        user_sequences = {}
        
        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id].sort_values('timestamp')
            items = user_df['item_id'].values
            
            sequences = []
            for i in range(len(items) - seq_length):
                sequence = items[i:i + seq_length]
                sequences.append(sequence)
            
            if sequences:
                user_sequences[user_id] = sequences
        
        print(f"✓ Created temporal sequences for {len(user_sequences)} users")
        
        return user_sequences
    
    def get_temporal_context(self, 
                            df: pd.DataFrame,
                            user_id: int,
                            item_id: int) -> Dict:
        """
        Get temporal context features for a user-item pair.
        
        Args:
            df: Input dataframe
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Dictionary with temporal features
        """
        user_df = df[df['user_id'] == user_id].sort_values('timestamp')
        
        # Time since last interaction
        if len(user_df) > 0:
            last_interaction_time = user_df['timestamp_seconds'].iloc[-1]
            current_time = user_df['timestamp_seconds'].iloc[-1]
            time_since_last = current_time - last_interaction_time
        else:
            time_since_last = 0
        
        # User activity in temporal window
        recent_threshold = current_time - self.temporal_window
        recent_interactions = len(user_df[user_df['timestamp_seconds'] > recent_threshold])
        
        return {
            'time_since_last_interaction': time_since_last,
            'recent_interaction_count': recent_interactions
        }


class DataLoader:
    """Load and batch data for training."""
    
    def __init__(self, batch_size: int = 32):
        """Initialize data loader."""
        self.batch_size = batch_size
        
    def get_batches(self, df: pd.DataFrame):
        """Generate batches from dataframe."""
        n_batches = len(df) // self.batch_size
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batch = df.iloc[start_idx:end_idx]
            
            yield {
                'user_ids': batch['user_id'].values,
                'item_ids': batch['item_id'].values,
                'ratings': batch['rating'].values,
                'timestamps': batch['timestamp_seconds'].values
            }
    
    def get_user_item_pairs(self, df: pd.DataFrame):
        """Get user-item pairs with ratings and timestamps."""
        pairs = []
        for _, row in df.iterrows():
            pairs.append({
                'user_id': row['user_id'],
                'item_id': row['item_id'],
                'rating': row['rating'],
                'timestamp': row['timestamp_seconds']
            })
        return pairs


def save_data(df: pd.DataFrame, filepath: str):
    """Save dataframe to CSV."""
    df.to_csv(filepath, index=False)
    print(f"✓ Data saved to {filepath}")


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataframe from CSV."""
    df = pd.read_csv(filepath)
    print(f"✓ Data loaded from {filepath}")
    return df


if __name__ == "__main__":
    # Example usage
    print("=== Time-Aware Recommender System: Data Generation ===\n")
    
    # Generate data
    generator = DataGenerator(n_users=500, n_items=300, n_interactions=30000)
    df = generator.generate(days=365)
    generator.add_item_features(df, n_features=10)
    generator.add_user_features(df, n_features=5)
    
    # Process data
    processor = DataProcessor()
    train_df, test_df = processor.create_train_test_split(df, temporal_split=True)
    
    # Create sequences
    user_sequences = processor.create_temporal_sequences(train_df, seq_length=5)
    
    # Create matrices
    user_item_matrix = processor.create_user_item_matrix(train_df)
    print(f"✓ User-item matrix shape: {user_item_matrix.shape}")
    
    # Save data
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    save_data(df, str(data_dir / "interactions.csv"))
    save_data(train_df, str(data_dir / "train_interactions.csv"))
    save_data(test_df, str(data_dir / "test_interactions.csv"))
    
    print("\n✓ Data generation and processing complete!")
