from keras import Model
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import keras.backend as K
from dataset.utils.shared import checkpoint_url

K.clear_session()
x = InceptionResNetV2(include_top=True, weights='imagenet')
# load_model(checkpoint_url)
y = x.output
model = Model(inputs=x.input, outputs=y)


def inception_resnet_v2_predict(images):
    images = images.astype(np.float32)
    predictions = model.predict(preprocess_input(images))
    return predictions
