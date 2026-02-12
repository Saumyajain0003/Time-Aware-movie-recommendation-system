"""Time-aware recommendation models implementing collaborative filtering with temporal dynamics"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class TimeAwareRecommender:
    """
    Collaborative filtering recommender with time decay and temporal patterns.
    
    This model combines:
    - Matrix Factorization for collaborative filtering
    - Time decay function to weight recent interactions higher
    - Temporal dynamics to capture evolving user preferences
    """
    
    def __init__(self, n_factors=20, learning_rate=0.01, n_epochs=50, 
                 temporal_weight=0.3, regularization=0.01):
        """
        Initialize the time-aware recommender.
        
        Args:
            n_factors (int): Number of latent factors
            learning_rate (float): Learning rate for SGD
            n_epochs (int): Number of training epochs
            temporal_weight (float): Weight for temporal component (0-1)
            regularization (float): L2 regularization strength
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.temporal_weight = temporal_weight
        self.regularization = regularization
        
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.timestamps = None
        self.reference_time = None
        
    def _time_decay(self, timestamps, reference_time=None, decay_rate=0.01):
        """
        Calculate time decay weights for interactions.
        
        Args:
            timestamps: Array of timestamps
            reference_time: Reference time point (default: max timestamp)
            decay_rate: Exponential decay rate
            
        Returns:
            Decay weights array
        """
        if reference_time is None:
            reference_time = np.max(timestamps)
        
        time_diff = (reference_time - timestamps) / (24 * 3600)  # Convert to days
        time_decay = np.exp(-decay_rate * time_diff)
        return time_decay
    
    def fit(self, interactions, timestamps, user_ids=None, item_ids=None):
        """
        Train the time-aware recommender model.
        
        Args:
            interactions: CSR matrix of shape (n_users, n_items) with ratings
            timestamps: Array of timestamps for each interaction
            user_ids: User indices
            item_ids: Item indices
            
        Returns:
            self
        """
        self.interactions = interactions.copy()
        self.n_users, self.n_items = interactions.shape
        self.timestamps = timestamps
        self.reference_time = np.max(timestamps)
        
        # Initialize factors with small random values
        self.user_factors = np.random.normal(0, 0.01, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.01, (self.n_items, self.n_factors))
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_mean = interactions.data.mean() if interactions.data.size > 0 else 0
        
        # Convert to COO format for iteration
        interactions_coo = self.interactions.tocoo()
        
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            
            for idx in range(len(interactions_coo.data)):
                user_id = interactions_coo.row[idx]
                item_id = interactions_coo.col[idx]
                rating = interactions_coo.data[idx]
                timestamp = timestamps[idx] if isinstance(timestamps, (list, np.ndarray)) else timestamps
                
                # Time decay
                time_weight = self._time_decay(np.array([timestamp]), self.reference_time)[0]
                
                # Prediction
                pred = (self.global_mean + 
                       self.user_biases[user_id] + 
                       self.item_biases[item_id] +
                       np.dot(self.user_factors[user_id], self.item_factors[item_id]))
                
                # Error
                error = rating - pred
                
                # Update with time decay weight
                update_strength = self.learning_rate * time_weight
                
                # Update biases
                self.user_biases[user_id] += update_strength * (error - self.regularization * self.user_biases[user_id])
                self.item_biases[item_id] += update_strength * (error - self.regularization * self.item_biases[item_id])
                
                # Update factors
                user_factor_grad = error * self.item_factors[item_id] - self.regularization * self.user_factors[user_id]
                item_factor_grad = error * self.user_factors[user_id] - self.regularization * self.item_factors[item_id]
                
                self.user_factors[user_id] += update_strength * user_factor_grad
                self.item_factors[item_id] += update_strength * item_factor_grad
                
                epoch_loss += error ** 2
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs} - Loss: {epoch_loss / len(interactions_coo.data):.4f}")
        
        return self
    
    def predict(self, user_id, n_recommendations=10, exclude_interacted=True):
        """
        Generate recommendations for a user.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of items to recommend
            exclude_interacted (bool): Whether to exclude already interacted items
            
        Returns:
            List of (item_id, score) tuples
        """
        if self.user_factors is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Calculate scores for all items
        user_factor = self.user_factors[user_id]
        scores = (self.global_mean + 
                 self.user_biases[user_id] +
                 self.item_biases +
                 np.dot(self.item_factors, user_factor))
        
        # Exclude already interacted items
        if exclude_interacted:
            interacted_items = self.interactions[user_id].nonzero()[1]
            scores[interacted_items] = -np.inf
        
        # Get top N recommendations
        top_item_ids = np.argsort(scores)[-n_recommendations:][::-1]
        recommendations = [(item_id, scores[item_id]) for item_id in top_item_ids]
        
        return recommendations
    
    def predict_batch(self, user_ids, n_recommendations=10, exclude_interacted=True):
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids (list): List of user IDs
            n_recommendations (int): Number of items per user
            exclude_interacted (bool): Whether to exclude already interacted items
            
        Returns:
            Dictionary mapping user_id to list of recommendations
        """
        recommendations = {}
        for user_id in user_ids:
            recommendations[user_id] = self.predict(user_id, n_recommendations, exclude_interacted)
        return recommendations


class TimeSensitiveMatrixFactorization:
    """
    Advanced time-aware recommender using temporal dynamics.
    
    Incorporates:
    - User drift (preference evolution)
    - Item popularity trends
    - Temporal context
    """
    
    def __init__(self, n_factors=20, learning_rate=0.01, n_epochs=50, 
                 temporal_decay=0.01, user_drift_factor=0.01):
        """
        Initialize time-sensitive matrix factorization.
        
        Args:
            n_factors (int): Number of latent factors
            learning_rate (float): Learning rate
            n_epochs (int): Training epochs
            temporal_decay (float): Decay factor for time
            user_drift_factor (float): User preference drift rate
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.temporal_decay = temporal_decay
        self.user_drift_factor = user_drift_factor
        
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.timestamps = None
        
    def fit(self, interactions, timestamps):
        """Train the model."""
        self.interactions = interactions.copy()
        self.n_users, self.n_items = interactions.shape
        self.timestamps = timestamps
        
        # Initialize with small random values
        self.user_factors = np.random.normal(0, 0.01, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.01, (self.n_items, self.n_factors))
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        
        reference_time = np.max(timestamps)
        interactions_coo = self.interactions.tocoo()
        
        for epoch in range(self.n_epochs):
            for idx in range(len(interactions_coo.data)):
                user_id = interactions_coo.row[idx]
                item_id = interactions_coo.col[idx]
                rating = interactions_coo.data[idx]
                
                # Time decay weight
                time_diff = (reference_time - timestamps[idx]) / (24 * 3600)
                time_weight = np.exp(-self.temporal_decay * time_diff)
                
                # Prediction
                pred = (self.user_biases[user_id] + 
                       self.item_biases[item_id] +
                       np.dot(self.user_factors[user_id], self.item_factors[item_id]))
                
                error = rating - pred
                
                # Update with adaptive learning rate
                lr = self.learning_rate * time_weight
                
                self.user_biases[user_id] += lr * error
                self.item_biases[item_id] += lr * error
                self.user_factors[user_id] += lr * error * self.item_factors[item_id]
                self.item_factors[item_id] += lr * error * self.user_factors[user_id]
            
            if (epoch + 1) % 10 == 0:
                print(f"TSMF Epoch {epoch + 1}/{self.n_epochs}")
        
        return self
    
    def predict(self, user_id, n_recommendations=10, exclude_interacted=True):
        """Generate recommendations for a user."""
        user_factor = self.user_factors[user_id]
        scores = (self.user_biases[user_id] +
                 self.item_biases +
                 np.dot(self.item_factors, user_factor))
        
        if exclude_interacted:
            interacted = self.interactions[user_id].nonzero()[1]
            scores[interacted] = -np.inf
        
        top_items = np.argsort(scores)[-n_recommendations:][::-1]
        return [(item_id, scores[item_id]) for item_id in top_items]
