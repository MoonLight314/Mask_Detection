import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler


BATCH_SIZE = 32
DROP_OUT_RATE = 0.2

tf.get_logger().setLevel('ERROR')

dataset_info = pd.read_csv("meta_data.csv")


data_file_path = dataset_info[['file_path' , 'xmin' , 'ymin' , 'xmax' , 'ymax']]
mask = dataset_info['mask'].tolist()


le = LabelEncoder()
le.fit(mask)
print(le.classes_)

le_mask = le.transform(mask)
mask = tf.keras.utils.to_categorical(le_mask , num_classes=2)


file_path_train, file_path_val, y_train, y_val = train_test_split(data_file_path, mask, 
                                                                  test_size=0.25, 
                                                                  random_state=777, 
                                                                  stratify = mask)


print( len(file_path_train) , len(y_train) , len(file_path_val) , len(y_val) )



train_left = file_path_train['xmin'].tolist()
train_right = file_path_train['xmax'].tolist()
train_top = file_path_train['ymin'].tolist()
train_bottom = file_path_train['ymax'].tolist()
file_path_train = file_path_train['file_path'].tolist()



val_left = file_path_val['xmin'].tolist()
val_right = file_path_val['xmax'].tolist()
val_top = file_path_val['ymin'].tolist()
val_bottom = file_path_val['ymax'].tolist()
file_path_val = file_path_val['file_path'].tolist()

"""
for idx,path in enumerate(file_path_train):
    file_path_train[idx] = os.path.join( "./Dataset/Face_Mask_Detection_Dataset_Kaggle/images/" , path )

for idx,path in enumerate(file_path_val):
    file_path_val[idx] = os.path.join( "./Dataset/Face_Mask_Detection_Dataset_Kaggle/images/" , path )
"""



def load_image( image_path , left , right , top , bottom , label ):
    img = tf.io.read_file(image_path)
    #img = tf.image.decode_jpeg(img, channels=3)   
    img = tf.image.decode_png(img, channels=3)   
    img = tf.image.crop_to_bounding_box( img , top , left, bottom - top , right - left )

    """
    output_image = tf.image.encode_png(img)
    file_name = tf.constant('./Ouput_image.png')
    file = tf.io.write_file(file_name, output_image)    
    """
    
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)    
    
    return img , label



train_dataset = tf.data.Dataset.from_tensor_slices( (file_path_train , 
                                                     train_left , 
                                                     train_right , 
                                                     train_top , 
                                                     train_bottom , 
                                                     y_train) )

val_dataset = tf.data.Dataset.from_tensor_slices( (file_path_val , 
                                                   val_left , 
                                                   val_right , 
                                                   val_top , 
                                                   val_bottom ,
                                                   y_val) )



train_dataset = train_dataset.shuffle(buffer_size=len(file_path_train))\
                                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                                .repeat()\
                                .batch(BATCH_SIZE)\
                                .prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_dataset.shuffle(buffer_size=len(file_path_val))\
                            .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                            .repeat()\
                            .batch(BATCH_SIZE)\
                            .prefetch(tf.data.experimental.AUTOTUNE)    #



ResNet50 = tf.keras.applications.resnet.ResNet50(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)



model= Sequential()

model.add( ResNet50 )

model.add( GlobalAveragePooling2D() ) 
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 
model.add( Dense(128, activation='relu') )
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 

model.add( Dense(2, activation='softmax') )



initial_learning_rate = 0.01

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * np.math.exp(-k*epoch)

lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)



log_dir = os.path.join('Logs')
CHECKPOINT_PATH = os.path.join('CheckPoints_Mask_Detection')
tb_callback = TensorBoard(log_dir=log_dir)

cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                     monitor='val_accuracy',                     
                     save_best_only = True,
                     verbose = 1)




model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    #loss='binary_crossentropy',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


hist = model.fit(train_dataset,
                 validation_data=val_dataset,
                 callbacks=[lr_scheduler , cp , tb_callback],
                 steps_per_epoch = 200,
                 validation_steps = 50,
                 epochs = 20,
                 verbose = 1 
)