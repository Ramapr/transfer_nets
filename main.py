import gc
import sys
import h5py as h5
import numpy as np
from os.path import join, exists
from os import listdir
from os import mkdir
import pandas as pd
import sys
import gc
import pickle

from keras.models import load_model
import keras.backend as K
from keras.models import Model
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import f1_score, confusion_matrix, precision_score
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score


SourceFileLoader("models", "/home/models.py").load_module()
from models import vgg, dens121, mbn, resnet, xcp, nas, incp


#%%%

path2dset = "/data"
part = 'list_verh_'
tr = 'train.hdf5'
ts =  'test.hdf5'
_path = '/data/test'  

dir_ = {'mbn': mbn, 
        'res': resnet, 
        'xcp':xcp, 
        'incp':incp,
        'dens121': dens121,
        'nas':nas, 
        'vgg': vgg
       }

inp_shape = (192, 256, 3)
n_class = 4 

LR = 0.001
EPOCH = 20 # 30
BS = 64
pred_n = 10
N = 5

batch_sched = [10, 6, 4] #, 16, 16]
lr_sched = [.001, .0005, .0001] #, .00001]

clmn = ['name', 'n_iter', 'file', 'scikit_f1', 'cust_f1', 'AP', 'roc_auc', 'loss', 'prec', 'recall'] #### FIX




def get_callback_list(amodel, ModelFileNameMask):
  lr_scheduler = keras_utils_omilos.get_lr_scheduler(lr_sched, batch_sched)
  lr_logger = keras_utils_omilos.get_lr_logger(amodel)
  model_checkpointer = keras_utils_omilos.get_model_checkpointer(ModelFileNameMask)
  return [lr_scheduler, lr_logger, model_checkpointer]

#%%


tr_file = h5.File(join(path2dset, part + tr), 'r')
tr_img = tr_file['Aug_array'][...] / 255.
tr_y = tr_file['TAG_Aug_array'][...]
tr_file.close()

ts_file = h5.File(join(path2dset, part + ts), 'r')
ts_img = ts_file['Aug_array'][...] / 255.
ts_y = ts_file['TAG_Aug_array'][...]
ts_file.close()

mean_imgtr = np.mean(tr_img, axis=0)
mean_imgts = np.mean(ts_img, axis=0)


X_train, X_test = tr_img - mean_imgtr, ts_img - mean_imgts
y_train, y_test = tr_y, ts_y

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#%%
for nm, dl in dir_.items():
  # create folder
  if not exists(join(_path, nm)):
    try:
      mkdir(join(_path, nm))
    except OSError as e:
      print('e')
      sys.stdout.flush()

  model_path = join(_path, nm)
  print(nm)
  
  log = pd.DataFrame([], columns=clmn)
  mdl_iter = 0
  sys.stdout.flush()

  for n in range(N):
    K.clear_session()        
    mdl = dl(inp_shape, 4, 'softmax')
    opt = Adam(lr=LR, decay=1e-6)
    
    mdl.compile(optimizer=opt, 
                loss=categorical_crossentropy, 
                metrics=[f1])
    
    net_name = '{0}_{1}'.format(nm, n)
    name = net_name + ".{epoch:03d}-{loss:.4f}-{val_loss:.4f}-{f1:.4f}-{val_f1:.4f}.h5"
    FileName = join(model_path, name)
    callback_list = get_callback_list(mdl, FileName)    
    model_hist = mdl.fit(X_train,
                         y_train,
                         epochs=EPOCH,
                         batch_size=BS,
                         validation_data=(X_test, y_test),
                         verbose=2,
                         shuffle=True,
                         callbacks=callback_list)

    with open(join(model_path, net_name + '_hist.pkl'), 'wb') as f:
      pickle.dump(model_hist.history, f)
    del mdl
    gc.collect()
    
    ########### EVAL
    
    files = list(filter(lambda x: True if x.endswith('.h5') and x.startswith(net_name) else False, listdir(model_path)))
    f1v = np.asarray([float(f.split('-')[-1][:-3]) for f in files])
    top5 = f1v.argsort()[-pred_n:]
    f2pred = np.asarray(files)[top5]
    for f in f2pred:
      K.clear_session()
      if nm != 'nas':
        model = load_model(join(model_path, f), custom_objects={'f1':f1})
      else:
        model = load_model(join(model_path, f), custom_objects={'f1':f1}, compile=False) 
      
      yprrrrr = model.predict(X_test)
        
      y_pred = np.round(yprrrrr).astype(np.uint8)
      del model
      gc.collect()
      print(f)
      #print(model.summary())
      sys.stdout.flush()
      scf1 = f1_score(y_test, y_pred, average='weighted')
      #sys.stdout.flush()    
      cf1 = K.get_value(f1(y_test, y_pred))
      #sys.stdout.flush()
      #f1m = K.get_value(f1_m(y_test, y_pred))
      #sys.stdout.flush()
      AP = average_precision_score(y_test, y_pred)
      roc_auc = roc_auc_score(y_test, y_pred)
      #ca = K.get_value(categorical_crossentropy(convert_to_tensor(y_test, np.float32),
      #                                          convert_to_tensor(y_pred, np.float32)))
      #cc = np.mean(ca)
      cc = np.mean(K.get_value(categorical_crossentropy(convert_to_tensor(y_test, np.float32), convert_to_tensor(yprrrrr, np.float32))))
      print(cc)
      sys.stdout.flush()
      
      pre = precision_score(y_test, y_pred, average='weighted')
      rec = recall_score(y_test, y_pred, average='weighted')
      
      print("scf1 - {0}\ncuf11 - {1}".format(scf1, cf1)) #, f1m))
      sys.stdout.flush()

      pred = np.argmax(y_pred, 1)
      ref = np.argmax(y_test, 1)
      cm = confusion_matrix(ref, pred)
      np.save(join(model_path, f[:-3] + '_cm.npy' ), cm)
      log.loc[mdl_iter] = [nm, n, f, scf1, cf1, AP, roc_auc, cc, pre, rec]
      mdl_iter += 1
      log.to_hdf(join(model_path, 'log{0}.h5'.format(mdl_iter)), key='df', mode='w')