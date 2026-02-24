from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

def build_pca_ridge(n_components: int = 3, alpha: float = 316.22776601683796):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("model", Ridge(alpha=alpha))
    ])