import numpy as np
import pandas as pd
import os
import glob
import cv2
from tqdm import tqdm
import tensorflow as tf
import xml.etree.ElementTree as et

MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9
MARGIN_RATIO = 0.2




def save_file_fist():
    dir_path = "./Dataset/Face-Mask-Detection-master"
    data_file_path = []

    for (root, directories, files) in tqdm(os.walk(dir_path)):
        for file in files:
            file_path = os.path.join(root, file)
            data_file_path.append( file_path )


    #print( len(data_file_path) )

    #meta_data = pd.DataFrame( data_file_path , columns=['file_path'])
    #meta_data.to_csv("meta_data_Face-Mask-Detection-master.csv",index=False)
    return data_file_path







def get_face_coor_info(file_path):
    
    pass_file_list = []
    lefts = []
    tops = []
    rights = []
    bottoms = []
    masks = []
    low_confidence_cnt = 0

    net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )

    for file in tqdm(file_path):

        try:
            img = cv2.imread(file)
            rows, cols, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1.0)

            net.setInput(blob)
            detections = net.forward()

            detection = detections[0, 0]    
            i = np.argmax(detection[:,2])

            if i != 0:
                print(file , "Max index is not 0")
                continue

            if detection[i,2] < CONFIDENCE_FACE:
                #print(file , "Low CONFIDENCE_FACE" , detection[i,2])
                low_confidence_cnt += 1
                continue
            

            left = detection[i,3] * cols
            top = detection[i,4] * rows
            right = detection[i,5] * cols            
            bottom = detection[i,6] * rows

            left = int(left - int((right - left) * MARGIN_RATIO))
            top = int(top - int((bottom - top) * MARGIN_RATIO))
            right = int(right + int((right - left) * MARGIN_RATIO))
            bottom = int(bottom + int((bottom - top) * MARGIN_RATIO / 2))

            if left < 0:
                left = 0

            if right > cols:
                right = cols

            if top < 0:
                top = 0

            if bottom > rows:
                bottom = rows

            pass_file_list.append(file)
            lefts.append(left)
            tops.append(top)
            rights.append(right)
            bottoms.append(bottom)

            if "with_mask" in file:
                masks.append("with_mask")
            elif "without_mask" in file:
                masks.append("without_mask")
        
        except:
            print(file , " Error")


    print(len(pass_file_list))
    print("No. of Low Confidence : ",low_confidence_cnt)

    result = pd.DataFrame(list(zip(pass_file_list, masks , lefts , tops , rights , bottoms)), columns=['file_path','mask','xmin','ymin','xmax','ymax'])

    #result.to_csv("meta_data_#1.csv",index=False)

    result = result.astype({    'xmin':'int32', 
                                'ymin':'int32',
                                'xmax':'int32', 
                                'ymax':'int32',
                                })

    return result







def preprocessing_Face_Mask_Detection_Dataset_Kaggle():
    dir_path = "./Dataset/Face_Mask_Detection_Dataset_Kaggle/annotations/"
    image_dir_path = "./Dataset/Face_Mask_Detection_Dataset_Kaggle/images/"
    data_file_path = []

    for (root, directories, files) in tqdm(os.walk(dir_path)):
        for file in files:
            if '.xml' in file:
                file_path = os.path.join(root, file)
                data_file_path.append( file_path )



    meta_data = pd.DataFrame({"file_path":[], 
                            "mask":[],
                            "xmin":[],
                            "ymin":[],
                            "xmax":[],
                            "ymax":[]
                            })


    for path in tqdm(data_file_path):

        xtree=et.parse( path )
        xroot=xtree.getroot()

        mask_flag = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []

        for node in xroot:
            
            if node.tag == 'filename':
                fname = os.path.join(image_dir_path , node.text)

            if node.tag == 'object':
                name = node.find("name")
                mask_flag.append( name.text )

                box = node.find("bndbox")
                
                t = box.find("xmin")
                if t != None:
                    xmin.append( t.text )

                t = box.find("ymin")
                if t != None:
                    ymin.append( t.text )

                t = box.find("xmax")
                if t != None:
                    xmax.append( t.text )

                t = box.find("ymax")
                if t != None:
                    ymax.append( t.text )
                


        file_name = [fname] * len(xmin)

        tmp = pd.DataFrame({"file_path":file_name , 
                            "mask":mask_flag,
                            "xmin":xmin,
                            "ymin":ymin,
                            "xmax":xmax,
                            "ymax":ymax
                            })

        meta_data = pd.concat( [meta_data,tmp] )

    #print('End')
    #meta_data.to_csv("meta_data_#2.csv",index=False)    

    meta_data = meta_data.astype({  'xmin':'int32', 
                                    'ymin':'int32',
                                    'xmax':'int32', 
                                    'ymax':'int32',
                                    })

    return meta_data









def verify_image_file(meta_data):

    #meta_data = pd.read_csv("merged_meta_data_211209_Rev_01.csv")

    train_left = meta_data['xmin'].tolist()
    train_right = meta_data['xmax'].tolist()
    train_top = meta_data['ymin'].tolist()
    train_bottom = meta_data['ymax'].tolist()
    train_mask = meta_data['mask'].tolist()
    file_path_train = meta_data['file_path'].tolist()

    new_left = []
    new_right = []
    new_top = []
    new_bottom = []
    new_file_path = []
    new_mask = []

    for idx,image_path in tqdm(enumerate( file_path_train)):
        
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)   
            
            img = tf.image.crop_to_bounding_box( img , train_top[idx] , train_left[idx], train_bottom[idx] - train_top[idx] , train_right[idx] - train_left[idx] )

            """
            output_image = tf.image.encode_png(img)
            file_name = tf.constant('./Ouput_image.png')
            file = tf.io.write_file(file_name, output_image)    
            """
            
            img = tf.image.resize(img, (224, 224))
            img = tf.keras.applications.resnet50.preprocess_input(img)

            new_left.append(train_left[idx])
            new_right.append(train_right[idx])
            new_top.append(train_top[idx])
            new_bottom.append(train_bottom[idx])
            new_file_path.append(image_path)
            new_mask.append(train_mask[idx])
        
        except Exception as e:
            print(e)
            continue
    
    print(len(new_file_path))

    result = pd.DataFrame(list(zip(new_file_path, new_mask , new_left , new_top , new_right , new_bottom)), columns=['file_path','mask','xmin','ymin','xmax','ymax'])

    #result.to_csv("merged_meta_data_211209_Rev_02.csv",index=False)

    return result










if __name__== '__main__':    

    # 이 Code는 'Face-Mask-Detection-master' Dataset을 Train에 사용하기 위한 Pre-Processing용이다.

    # 가장 먼저 File List를 얻는다.
    data_file_path = save_file_fist()

    # 각 File에서 얼굴에 대한 정보를 추출합니다.
    # 얼굴을 추출하는 방법은 OpenCV에 제공하는 Tensorflow DNN Module, opencv_face_detector를 사용합니다.
    meta_data_01 = get_face_coor_info( data_file_path )
    #meta_data_01.to_csv("meta_data_01.csv",index=False)

    # 이 Code는 'Face_Mask_Detection_Dataset_Kaggle' Dataset을 Train에 사용하기 위한 Pre-Processing용이다.
    meta_data_02 = preprocessing_Face_Mask_Detection_Dataset_Kaggle()
    #meta_data_02.to_csv("meta_data_02.csv",index=False)

    meta_data = pd.concat([meta_data_01 , meta_data_02])
    
    #meta_data.to_csv("meta_data.csv",index=False)
    #meta_data = pd.read_csv("meta_data.csv")

    meta_data = verify_image_file(meta_data)

    meta_data = meta_data.replace({'mask':'mask_weared_incorrect'},'with_mask')
    meta_data.to_csv("meta_data.csv",index=False)