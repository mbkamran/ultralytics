# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK

"""TODO: Need to edit 

if get_model():
    ...
    model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)

^^ replace OBBModel with DetectFBModel + have to write class for that first

"""

class DetectFBTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a DetectFB model.

    Example:
        ```python
        from ultralytics.models.yolo.detectfb import DetectFBTrainer

        args = dict(model="yolo11n.pt", data="dota8.yaml", epochs=3)
        trainer = DetectFBTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a DetectFBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "detectfb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return DetectFBModel initialized with specified config and weights."""
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
