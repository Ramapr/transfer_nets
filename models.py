# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:35:56 2020

@author: Mi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:39:22 2020

@author: Mi


код для создания и импорта моделей 

"""

from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet169 #, DenseNet201
from keras.applications.nasnet import NASNetMobile
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

from keras.layers import GlobalAveragePooling2D, Dense, Flatten
from keras import Model


#%%

def vgg(inp_shape, out, out_act, wpath=None):
  vgg = VGG16(input_shape=inp_shape, include_top=False, weights=None, input_tensor=None)
  x = GlobalAveragePooling2D()(vgg.output)
  x = Dense(512, activation='relu')(x)
  x = Dense(64, activation='relu')(x)
  xout = Dense(out, activation=out_act)(x)  
  mdl = Model(vgg.input, xout)
  if not wpath:
    weights_path = "/data/Model/Additional/weights_pre/" + "vgg16_notop.h5"
    mdl.load_weights(weights_path, by_name=True)
  return mdl

#%%
  
def dens121(inp_shape, out, out_act, wpath=None):
  den = DenseNet121(input_shape=inp_shape, include_top=False, weights=None, input_tensor=None)
  x = GlobalAveragePooling2D()(den.output)
  xout = Dense(out, activation=out_act)(x)  
  mdl = Model(den.input, xout)
  if not wpath:
    weights_path = "/data/Model/Additional/weights_pre/" + 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    mdl.load_weights(weights_path, by_name=True)
  return mdl

#%%

#def dens169(wpath=None):
#  den = DenseNet169(input_shape=(224, 224, 3), include_top=False, weights=None, input_tensor=None)
#  x = GlobalAveragePooling2D()(den.output)
#  out = Dense(1, activation='sigmoid')(x)
#  mdl = Model(den.input, out)
#  if not wpath:
#    weights_path = "/data/NFP/Dev/Model/Additional/weights_pre/" + 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
#    mdl.load_weights(weights_path, by_name=True)
#  return mdl

#%%
  
def mbn(inp_shape, out, out_act, wpath=None):
  xcp = MobileNetV2(input_shape=inp_shape, include_top=False, weights=None, input_tensor=None)  
  x = GlobalAveragePooling2D()(xcp.output)
  xout = Dense(out, activation=out_act)(x)  
  mdl = Model(xcp.input, xout)
  if not wpath:
    weights_path = "/data/Model/Additional/weights_pre/" + "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    mdl.load_weights(weights_path, by_name=True)
  return mdl

#%%

def resnet(inp_shape, out, out_act, wpath=None):
  res = ResNet50(input_shape=inp_shape, include_top=False, weights=None, input_tensor=None)
  x = GlobalAveragePooling2D()(res.output)
  xout = Dense(out, activation=out_act)(x)  
  mdl = Model(res.input, xout)
  if not wpath:
    weights_path = "/data/Model/Additional/weights_pre/" + "resnet50.h5"
    mdl.load_weights(weights_path, by_name=True)
  return mdl

#%%
  
def incp(inp_shape, out, out_act, wpath=None):
  incp = InceptionV3(input_shape=inp_shape, include_top=False, weights=None, input_tensor=None)
  x = GlobalAveragePooling2D()(incp.output)
  xout = Dense(out, activation=out_act)(x)  
  mdl = Model(incp.input, xout)
  if not wpath:
    weights_path = "/data/Model/Additional/weights_pre/" + "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    mdl.load_weights(weights_path, by_name=True)
  return mdl
  
#%%
  
def xcp(inp_shape, out, out_act, wpath=None):
  xcp = Xception(input_shape=inp_shape, include_top=False, weights=None, input_tensor=None)
  x = GlobalAveragePooling2D()(xcp.output)
  xout = Dense(out, activation=out_act)(x)  
  mdl = Model(xcp.input, xout)
  if not wpath:
    weights_path = "/data/Model/Additional/weights_pre/" + "xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
    mdl.load_weights(weights_path, by_name=True)
  return mdl

#%%
  
def nas(inp_shape, out, out_act, wpath=None):
  net = NASNetMobile(input_shape=inp_shape, include_top=False, weights=None)
  x = GlobalAveragePooling2D()(net.output)
  xout = Dense(out, activation=out_act)(x)  
  mdl = Model(net.input, xout)
  if not wpath:
    weights_path = "/data/Model/Additional/weights_pre/" + "NASNet-mobile-no-top.h5"
    mdl.load_weights(weights_path, by_name=True)
  return mdl
    
#%%  
  