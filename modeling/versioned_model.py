class VersionedModel:
    """Wrapper for XGBoost model with versioning"""
    def __init__(self, model, version):
        self.model = model
        self.version = version
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save_model(path)
