# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectFBPredictor
from .train import DetectFBTrainer
from .val import DetectFBValidator

__all__ = "DetectFBPredictor", "DetectFBTrainer", "DetectFBValidator"
