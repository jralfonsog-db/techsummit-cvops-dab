import pytorch_lightning as pl
from torchvision import models
import torchvision.transforms as tf
from torchmetrics import Accuracy
import torch.nn.functional as F
import torchmetrics.functional as FM
import torch
import logging
import datetime as dt
from io import BytesIO
from PIL import Image
import mlflow


class CVModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        # instantiate model in evaluation mode
        model.to(torch.device("cpu"))
        self.model = model.eval()

    def feature_extractor(self, image):
        from PIL import Image
        transform = tf.Compose([
            tf.Resize(256),
            tf.CenterCrop(224),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        #return transform(Image.open(io.BytesIO(data)))
        return transform(Image.open(BytesIO(base64.b64decode(image))))

    def predict(self, context, images):
        with torch.set_grad_enabled(False):
          from PIL import Image
          id2label = {0: "normal", 1: "anomaly"}
          # add here check if this is a DataFrame 
          # if this is an image remove iterrows 
          pil_images = torch.stack([self.feature_extractor(row[0]) for _, row in images.iterrows()])
          #pil_images = images.map(lambda x: self.feature_extractor(x))
          pil_images = pil_images.to(torch.device("cpu"))
          outputs = self.model(pil_images)
          preds = torch.max(outputs, 1)[1]
          probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1]
          labels = [id2label[pred] for pred in preds.tolist()]

          return pd.DataFrame( data=dict(
            score=probs,
            label=preds,
            labelName=labels)
          )
