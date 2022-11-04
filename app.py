import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation,\
                                    MaxPool2D, UpSampling2D, concatenate,\
                                    Input, Conv2DTranspose, MaxPooling2D,\
                                    Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras import backend as K
from PIL import Image, ImageDraw, ImageFont

from model import build_unet



st.title('Deforestration Detection using U-NET')


st.image('./Images/forest.jpeg')
a = st.sidebar.title('Options')

uploaded_file = st.sidebar.file_uploader("Choose an image...")
st.markdown('The image uploaded is:') 
image= Image.open(uploaded_file)
st.image(image)

## Defining UNET





## Loading Model 
final_filters = 2048
path_to_load = "/Pretrained_Weights.h5"


model_1 = build_unet(input_shape=(512, 512, 3),
                    filters=[2 ** i for i in range(5, int(np.log2(final_filters) + 1))], # Amount of filters in U-Net arch.
                     batchnorm=False, transpose=False, dropout_flag=False)
model_1.load_weights(path_to_load)





# preprocessing images
image1 = Image.open(uploaded_file)
i1 = np.array(image1)
image_orig = image1
image = i1[np.newaxis, ...]
prediction = model_1.predict(image)# Using the Model_1 built earlier
prediction_class1 = np.copy(prediction[..., 0]) # Forest
prediction_class2 = np.copy(prediction[..., 1]) # Deforest
prediction[..., 0] = prediction_class2 # RED - Deforest
prediction[..., 1] = prediction_class1 # GREEN - Forest


st.image(prediction[0])



