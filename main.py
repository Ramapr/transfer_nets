
import numpy
import gc
import sys
from os import listdir
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import f1_score, confusion_matrix






#%%

LR = .001
EPOCH = 20 # 30
BS = 32

batch_sched = [10, 6, 4] #, 16, 16]
lr_sched = [.001, .0005, .0001] #, .00001]

def get_callback_list(amodel, ModelFileNameMask):
  lr_scheduler = keras_utils_omilos.get_lr_scheduler(lr_sched, batch_sched)
  lr_logger = keras_utils_omilos.get_lr_logger(amodel)
  model_checkpointer = keras_utils_omilos.get_model_checkpointer(ModelFileNameMask)
  return [lr_scheduler, lr_logger, model_checkpointer]

#%%

file = h5.File(data_path, 'r')
img = file['Aug_array'][...] / 255.
tag = file['TAG_Aug_array'][...]
train_arg = file['Aug_array_train_arg'][...]
test_arg = file['Aug_array_test_arg'][...]
file.close()

X_train , X_test = img[train_arg[:], :, :, :], img[test_arg[:], :, :, :]
y_train , y_test = tag[train_arg[:], :], tag[test_arg[:], :]


#%%

for model in liast_models:
    pass
    mdl = model()
    opt = Adam(lr=LR, decay=1e-6)
    mdl.compile(optimizer=opt, loss=binary_crossentropy, metrics=[f1, test_f1])

    del mdl
    gc.collect()    
    w = list(filter(lambda x: True if x.endswith('.h5') else False, listdir(model_path)))
    w.sort()
    print(w)
    sys.stdout.flush()
    for i in w[14:]:
        print(i)
        sys.stdout.flush()
        model = load_model(join(model_path, i), custom_objects={'f1':f1, 'test_f1':test_f1})
        y_pred = np.round(model.predict(X_test)) 
        
        print(f1_score(y_test, y_pred))
        sys.stdout.flush()

        print(K.get_value(f1(y_test, y_pred)))
        sys.stdout.flush()

        print(f1_score(y_test, y_pred, average='weighted'))
        sys.stdout.flush()
        
        print(K.get_value(test_f1(y_test, y_pred)))
        sys.stdout.flush()
        print(confusion_matrix(y_test, y_pred))
        sys.stdout.flush()
        
        del model
        gc.collect()

