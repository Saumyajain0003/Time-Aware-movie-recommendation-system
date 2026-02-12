"""Evaluation metrics for recommendation systems"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationMetrics:
    """Comprehensive evaluation metrics for recommender systems"""
    
    @staticmethod
    def recall_at_k(recommendations, ground_truth, k=10):
        """
        Calculate recall@k metric.
        
        Recall = (# of recommended items in ground truth) / (total # of ground truth items)
        
        Args:
            recommendations (list): List of recommended item IDs
            ground_truth (list): List of true item IDs
            k (int): Consider top k recommendations
            
        Returns:
            float: Recall score
        """
        rec_items = set(recommendations[:k])
        true_items = set(ground_truth)
        
        if len(true_items) == 0:
            return 0.0
        
        return len(rec_items & true_items) / len(true_items)
    
    @staticmethod
    def precision_at_k(recommendations, ground_truth, k=10):
        """
        Calculate precision@k metric.
        
        Precision = (# of recommended items in ground truth) / k
        
        Args:
            recommendations (list): List of recommended item IDs
            ground_truth (list): List of true item IDs
            k (int): Consider top k recommendations
            
        Returns:
            float: Precision score
        """
        rec_items = set(recommendations[:k])
        true_items = set(ground_truth)
        
        if k == 0:
            return 0.0
        
        return len(rec_items & true_items) / k
    
    @staticmethod
    def ndcg_at_k(recommendations, ground_truth, k=10):
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain).
        
        DCG@k = Σ(rel_i / log2(i+1)) where rel_i = 1 if item i is relevant
        NDCG@k = DCG@k / IDCG@k (ideal DCG)
        
        Args:
            recommendations (list): Ranked list of recommended item IDs
            ground_truth (list): List of true item IDs
            k (int): Consider top k recommendations
            
        Returns:
            float: NDCG score (0-1)
        """
        rec_items = recommendations[:k]
        true_items = set(ground_truth)
        
        # Calculate DCG
        dcg = 0.0
        for idx, item in enumerate(rec_items):
            if item in true_items:
                dcg += 1.0 / np.log2(idx + 2)  # +2 because log is base 2 and 1-indexed
        
        # Calculate ideal DCG
        idcg = 0.0
        num_relevant = min(len(true_items), k)
        for idx in range(num_relevant):
            idcg += 1.0 / np.log2(idx + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mean_average_precision(recommendations, ground_truth, k=10):
        """
        Calculate MAP@k (Mean Average Precision).
        
        AP = (1/min(m, k)) * Σ(precision@i * rel_i)
        where m = total relevant items, rel_i = 1 if item i is relevant
        
        Args:
            recommendations (list): Ranked list of recommended item IDs
            ground_truth (list): List of true item IDs
            k (int): Consider top k recommendations
            
        Returns:
            float: MAP score
        """
        rec_items = recommendations[:k]
        true_items = set(ground_truth)
        
        score = 0.0
        num_hits = 0.0
        
        for idx, item in enumerate(rec_items):
            if item in true_items:
                num_hits += 1.0
                score += num_hits / (idx + 1.0)
        
        if len(true_items) == 0:
            return 0.0
        
        return score / min(len(true_items), k)
    
    @staticmethod
    def rmse(predictions, ground_truth):
        """
        Calculate RMSE (Root Mean Squared Error).
        
        RMSE = sqrt(mean((predictions - ground_truth)^2))
        
        Args:
            predictions (array): Predicted ratings
            ground_truth (array): True ratings
            
        Returns:
            float: RMSE score
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground_truth must have same length")
        
        mse = np.mean((predictions - ground_truth) ** 2)
        return np.sqrt(mse)
    
    @staticmethod
    def mae(predictions, ground_truth):
        """
        Calculate MAE (Mean Absolute Error).
        
        MAE = mean(|predictions - ground_truth|)
        
        Args:
            predictions (array): Predicted ratings
            ground_truth (array): True ratings
            
        Returns:
            float: MAE score
        """
        return np.mean(np.abs(np.array(predictions) - np.array(ground_truth)))
    
    @staticmethod
    def coverage(recommendations, total_items):
        """
        Calculate catalog coverage.
        
        Coverage = (# of unique items recommended across all users) / total_items
        
        Args:
            recommendations (dict): Dict mapping user_id to list of recommended items
            total_items (int): Total number of items in catalog
            
        Returns:
            float: Coverage score (0-1)
        """
        recommended_items = set()
        for items in recommendations.values():
            if isinstance(items, list) and len(items) > 0:
                if isinstance(items[0], tuple):
                    recommended_items.update([item[0] for item in items])
                else:
                    recommended_items.update(items)
        
        if total_items == 0:
            return 0.0
        
        return len(recommended_items) / total_items
    
    @staticmethod
    def diversity(recommendations, item_features):
        """
        Calculate recommendation diversity using item feature similarity.
        
        Diversity = 1 - (average pairwise similarity of recommended items)
        
        Args:
            recommendations (list): List of recommended item IDs
            item_features (array): Item feature matrix (n_items x n_features)
            
        Returns:
            float: Diversity score (0-1)
        """
        if len(recommendations) < 2:
            return 1.0
        
        # Get features for recommended items
        rec_features = item_features[recommendations]
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(rec_features)
        
        # Average similarity (excluding diagonal)
        n = len(recommendations)
        total_sim = np.sum(similarities) - n  # Subtract diagonal (self-similarity = 1)
        avg_sim = total_sim / (n * (n - 1)) if n > 1 else 0
        
        return 1.0 - avg_sim
    
    @staticmethod
    def hr_at_k(recommendations, ground_truth, k=10):
        """
        Calculate Hit Rate@k (HR@k).
        
        HR = (# of users with at least one hit) / total users
        
        Args:
            recommendations (dict): Dict mapping user_id to recommended items
            ground_truth (dict): Dict mapping user_id to ground truth items
            k (int): Consider top k
            
        Returns:
            float: Hit rate
        """
        hits = 0
        for user_id, rec_items in recommendations.items():
            if user_id not in ground_truth:
                continue
            
            rec_set = set(rec_items[:k]) if isinstance(rec_items[0], (int, np.integer)) else {item[0] for item in rec_items[:k]}
            true_set = set(ground_truth[user_id])
            
            if len(rec_set & true_set) > 0:
                hits += 1
        
        if len(recommendations) == 0:
            return 0.0
        
        return hits / len(recommendations)
    
    @staticmethod
    def evaluate_model(recommendations_dict, ground_truth_dict, item_features=None, k_values=[5, 10, 20]):
        """
        Comprehensive evaluation of recommendation model.
        
        Args:
            recommendations_dict (dict): Dict mapping user_id to (item_id, score) tuples
            ground_truth_dict (dict): Dict mapping user_id to ground truth items
            item_features (array): Item feature matrix for diversity calculation
            k_values (list): K values for top-k metrics
            
        Returns:
            dict: Dictionary with all evaluation metrics
        """
        results = {}
        
        for k in k_values:
            recall_scores = []
            precision_scores = []
            ndcg_scores = []
            map_scores = []
            
            for user_id, recommendations in recommendations_dict.items():
                if user_id not in ground_truth_dict:
                    continue
                
                ground_truth = ground_truth_dict[user_id]
                if len(ground_truth) == 0:
                    continue
                
                # Extract item IDs from (item_id, score) tuples
                rec_items = [item[0] if isinstance(item, tuple) else item for item in recommendations]
                
                recall_scores.append(RecommendationMetrics.recall_at_k(rec_items, ground_truth, k))
                precision_scores.append(RecommendationMetrics.precision_at_k(rec_items, ground_truth, k))
                ndcg_scores.append(RecommendationMetrics.ndcg_at_k(rec_items, ground_truth, k))
                map_scores.append(RecommendationMetrics.mean_average_precision(rec_items, ground_truth, k))
            
            results[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
            results[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
            results[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
            results[f'map@{k}'] = np.mean(map_scores) if map_scores else 0.0
        
        # Coverage
        total_items = max(max(items) for items in ground_truth_dict.values() if items) + 1 if ground_truth_dict else 1
        results['coverage'] = RecommendationMetrics.coverage(recommendations_dict, total_items)
        
        # Diversity (if item features provided)
        if item_features is not None:
            diversity_scores = []
            for recs in recommendations_dict.values():
                rec_items = [item[0] if isinstance(item, tuple) else item for item in recs]
                diversity_scores.append(RecommendationMetrics.diversity(rec_items, item_features))
            results['diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # Hit Rate
        results['hr@10'] = RecommendationMetrics.hr_at_k(recommendations_dict, ground_truth_dict, 10)
        
        return results
