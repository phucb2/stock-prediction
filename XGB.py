from .SES import Predictor
import numpy as np
import xgboost as xgb

class XGBPredictor(Predictor):
    def __init__(self):
        super().__init__()

    def predict(self, P: np.ndarray, V: np.ndarray, h: int = 20) -> np.ndarray:
        return np.zeros(len(P))
    
__all__ = ["XGBPredictor"]