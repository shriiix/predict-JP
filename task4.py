import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

class FICOQuantizer:
    """
    Quantize FICO scores into rating buckets.
    Rating 1 = Highest risk (lowest FICO)
    Rating N = Lowest risk (highest FICO)
    """
    
    def __init__(self, method='mse', n_buckets=5):
        self.method = method
        self.n_buckets = n_buckets
        self.boundaries = None
        self.centroids = None
        self.fitted = False
    
    def fit(self, fico_scores, defaults=None):
        """Fit the quantizer to FICO score data."""
        fico_scores = np.array(fico_scores)
        
        if self.method == 'mse':
            self.boundaries, self.centroids = self._quantize_mse(fico_scores)
        elif self.method == 'log_likelihood':
            if defaults is None:
                raise ValueError("defaults required for log_likelihood method")
            self.boundaries = self._quantize_log_likelihood(fico_scores, np.array(defaults))
            # Calculate centroids after boundaries are set
            self.fitted = True  # Set this before calling transform
            self.centroids = self._calculate_centroids(fico_scores)
            self.fitted = False  # Reset temporarily
        else:
            raise ValueError("method must be 'mse' or 'log_likelihood'")
        
        self.fitted = True
        return self
    
    def _quantize_mse(self, fico_scores):
        """Quantize using K-means to minimize MSE."""
        X = fico_scores.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_buckets, random_state=42, n_init=10)
        kmeans.fit(X)
        
        centroids = sorted(kmeans.cluster_centers_.flatten())
        
        boundaries = [np.min(fico_scores)]
        for i in range(len(centroids) - 1):
            boundaries.append((centroids[i] + centroids[i + 1]) / 2)
        boundaries.append(np.max(fico_scores) + 1)
        
        return boundaries, centroids
    
    def _quantize_log_likelihood(self, fico_scores, defaults):
        """Quantize using equal-frequency buckets (approximates max log-likelihood)."""
        quantiles = np.linspace(0, 100, self.n_buckets + 1)
        boundaries = [fico_scores.min()]
        for q in quantiles[1:-1]:
            boundaries.append(np.percentile(fico_scores, q))
        boundaries.append(fico_scores.max() + 1)
        return sorted(set(boundaries))
    
    def _calculate_centroids(self, fico_scores):
        """Calculate centroid for each bucket."""
        centroids = []
        bucket_indices = np.digitize(fico_scores, self.boundaries) - 1
        for i in range(self.n_buckets):
            bucket_scores = fico_scores[bucket_indices == i]
            if len(bucket_scores) > 0:
                centroids.append(np.mean(bucket_scores))
        return centroids
    
    def transform(self, fico_scores):
        """
        Transform FICO scores to ratings.
        Returns ratings where 1 = worst credit, n_buckets = best credit
        """
        if not self.fitted:
            raise ValueError("Quantizer must be fitted before transform")
        
        fico_scores = np.array(fico_scores)
        bucket_indices = np.digitize(fico_scores, self.boundaries) - 1
        bucket_indices = np.clip(bucket_indices, 0, self.n_buckets - 1)
        
        # Convert to ratings (1 = worst/lowest FICO, n = best/highest FICO)
        ratings = bucket_indices + 1
        
        return ratings
    
    def get_rating_map(self):
        """Get a dictionary mapping FICO ranges to ratings."""
        rating_map = {}
        for i in range(self.n_buckets):
            rating_map[f"Rating {i+1}"] = {
                'FICO_Range': f"{self.boundaries[i]:.0f}-{self.boundaries[i+1]:.0f}",
                'Centroid': f"{self.centroids[i]:.1f}" if self.centroids else "N/A"
            }
        return rating_map
    
    def predict_rating(self, fico_score):
        """Predict rating for a single FICO score."""
        return self.transform(np.array([fico_score]))[0]


# EXAMPLE 1: MSE Method (K-Means Clustering)
print("="*80)
print("METHOD 1: MEAN SQUARED ERROR QUANTIZATION")
print("="*80)

quantizer_mse = FICOQuantizer(method='mse', n_buckets=5)
quantizer_mse.fit(df['fico_score'].values)

print("\n5-Bucket Rating Map (MSE Method):")
for rating, info in quantizer_mse.get_rating_map().items():
    print(f"  {rating}: {info['FICO_Range']} (centroid: {info['Centroid']})")

# Show performance
ratings_mse = quantizer_mse.transform(df['fico_score'].values)
print("\nPerformance by Rating:")
for rating in range(1, 6):
    mask = ratings_mse == rating
    n = np.sum(mask)
    if n > 0:
        defaults = np.sum(df[mask]['default'])
        pd_rate = defaults / n * 100
        avg_fico = df[mask]['fico_score'].mean()
        print(f"  Rating {rating}: n={n:4d}, avg_FICO={avg_fico:.0f}, "
              f"defaults={defaults:3d}, PD={pd_rate:.2f}%")


# EXAMPLE 2: Log-Likelihood Method (Equal Frequency)
print("\n" + "="*80)
print("METHOD 2: LOG-LIKELIHOOD QUANTIZATION")
print("="*80)

quantizer_ll = FICOQuantizer(method='log_likelihood', n_buckets=5)
quantizer_ll.fit(df['fico_score'].values, df['default'].values)

print("\n5-Bucket Rating Map (Log-Likelihood Method):")
for rating, info in quantizer_ll.get_rating_map().items():
    print(f"  {rating}: {info['FICO_Range']} (centroid: {info['Centroid']})")

ratings_ll = quantizer_ll.transform(df['fico_score'].values)
print("\nPerformance by Rating:")
for rating in range(1, 6):
    mask = ratings_ll == rating
    n = np.sum(mask)
    if n > 0:
        defaults = np.sum(df[mask]['default'])
        pd_rate = defaults / n * 100
        avg_fico = df[mask]['fico_score'].mean()
        print(f"  Rating {rating}: n={n:4d}, avg_FICO={avg_fico:.0f}, "
              f"defaults={defaults:3d}, PD={pd_rate:.2f}%")


# EXAMPLE 3: Using the Quantizer
print("\n" + "="*80)
print("PRACTICAL EXAMPLES")
print("="*80)

test_scores = [450, 550, 620, 680, 750]
print("\nFICO Score to Rating Mapping:")
for score in test_scores:
    rating_mse = quantizer_mse.predict_rating(score)
    rating_ll = quantizer_ll.predict_rating(score)
    print(f"  FICO {score}: Rating {rating_mse} (MSE), Rating {rating_ll} (Log-Likelihood)")


# EXAMPLE 4: 10-Bucket Version for More Granularity
print("\n" + "="*80)
print("10-BUCKET VERSION (MSE Method)")
print("="*80)

quantizer_10 = FICOQuantizer(method='mse', n_buckets=10)
quantizer_10.fit(df['fico_score'].values)

print("\n10-Bucket Rating Map:")
for rating, info in quantizer_10.get_rating_map().items():
    print(f"  {rating}: {info['FICO_Range']}")

# Save quantizer function
def create_fico_rating_function(quantizer):
    """
    Create a standalone function for production use.
    """
    boundaries = quantizer.boundaries
    n_buckets = quantizer.n_buckets
    
    def fico_to_rating(fico_score):
        """Convert FICO score to rating (1=worst, n=best)."""
        if fico_score < boundaries[0]:
            return 1
        if fico_score >= boundaries[-1]:
            return n_buckets
        
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= fico_score < boundaries[i+1]:
                return i + 1
        return n_buckets
    
    return fico_to_rating

# Create production function
fico_rater = create_fico_rating_function(quantizer_mse)

print("\n" + "="*80)
print("PRODUCTION FUNCTION READY")
print("="*80)
print("\nExample usage:")
print(f"  fico_rater(480) = {fico_rater(480)}")
print(f"  fico_rater(640) = {fico_rater(640)}")
print(f"  fico_rater(750) = {fico_rater(750)}")
