from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from config import *
from .chexnet import ChexNet
from .unet import Unet
from .heatmap import HeatmapGenerator
from .constant import IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from .utils import blend_segmentation
import argparse

import cv2
import json
import pydicom as dicom
import matplotlib.pyplot as plt
from copy import deepcopy
import glob
import os
import pandas as pd
from tqdm import tqdm
from skimage import io
from PIL import ImageFilter
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def threshold(minimum,maximum,image,binary=True):
    if(binary):
        image[image<minimum]=0
        image[image>maximum]=0
        image[(image>0)]=1
    else:
        image[image<minimum]=0
        image[image>maximum]=0
        
    return image

def get_1D_coord(contours):
    
    global_list=[]
    for contour_id in range(0,len(contours)):
        local_list=[]
        for point_idx in range(0,contours[contour_id].shape[0]):
            if(point_idx==0):
                X_0= contours[contour_id][point_idx][0][0]
                Y_0 = contours[contour_id][point_idx][0][1]

            X = contours[contour_id][point_idx][0][0]
            Y = contours[contour_id][point_idx][0][1]
            local_list.append(X)
            local_list.append(Y)

            if(point_idx == contours[contour_id].shape[0]-1 ):
                local_list.append(X_0)
                local_list.append(Y_0)


        global_list.append(deepcopy(local_list))
        
    return(global_list)


def get_pairs(cont_new):
    pairs=[]
    for i in range(0,len(cont_new)):
        if (i<(len(cont_new)-1)):
            pairs.append((cont_new[i],cont_new[i+1]))
        else:
            pairs.append((cont_new[i],cont_new[0]))
    
    return(pairs)



def get_coord_dict(cont_new):
    final_pairs = get_pairs(cont_new)
    
    line_list = []
    for i in range(0,len(final_pairs)):
        line_list.append(final_pairs[i][0])
    
    return line_list



def get_1D_coord(contours):
    
    global_list=[]
    for contour_id in range(0,len(contours)):
        local_list=[]
        for point_idx in range(0,contours[contour_id].shape[0]):
            if(point_idx==0):
                X_0= contours[contour_id][point_idx][0][0].astype('float')
                Y_0 = contours[contour_id][point_idx][0][1].astype('float')

            X = contours[contour_id][point_idx][0][0].astype('float')
            Y = contours[contour_id][point_idx][0][1].astype('float')
            local_list.append(X)
            local_list.append(Y)

            if(point_idx == contours[contour_id].shape[0]-1):
                local_list.append(X_0)
                local_list.append(Y_0)


        global_list.append(deepcopy(local_list))
        
    return(global_list)




def get_json(data,coord,input_f):
    
    
    '''
    data: original json data format file
    coord: 1D coordinates with alternate x and y 
    input_f:dicom file path corresponding to the 'coord'
    
    returns: edited data['allTools'][0]
    
    '''
    
#     print('========coord = ',coord)
    
    final_coord = [[coord[x],coord[y]] for x,y in zip(list(range(0,len(coord),2)),list(range(1,len(coord),2)))]
#     print('final_coord_1 = ',final_coord)
    
#     print('final coord = ',final_coord)
    
    lines_dictionary = get_coord_dict(final_coord)
#     print('lines_dict = ',lines_dictionary)
    dcm_file = dicom.dcmread(input_f)
    studyInstanceUid = dcm_file.StudyInstanceUID
    seriesInstanceUid = dcm_file.SeriesInstanceUID
    sopInstanceUid = dcm_file.SOPInstanceUID

    data = {
        'type': 'Freehand',
        'StudyInstanceUID': studyInstanceUid,
        'SeriesInstanceUID': seriesInstanceUid,
        'SOPInstanceUID': sopInstanceUid,
        'points': lines_dictionary,
        'Finding_name': 'Abnormality'
    }

    return(data)
    

def save_overlayed_heatmap(heatmap,xray,parent_dir='.',quality_jpeg=50,want_alpha=False,transparency=0.3,threshold_heatmap=False,threshold_min=150):
    cm = plt.get_cmap('jet')
    if(threshold_heatmap):
        heatmap=threshold(threshold_min,heatmap.max(),heatmap,False)

    heatmap = np.array(Image.fromarray(heatmap).filter(ImageFilter.GaussianBlur(100)))

    background = Image.fromarray(xray)
    overlay = Image.fromarray((cm(heatmap)[:, :, :3] * 255).astype(np.uint8))

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    new_img = Image.blend(background, overlay,transparency)

    if (want_alpha):
        new_img.save(parent_dir+file_name.replace('.jpg','.png'))
    else:



        new_img_array=np.array(new_img)[:,:,0:3]
        io.imsave(parent_dir+".jpg",new_img_array,quality=quality_jpeg)
    # cv2.imwrite(parent_dir+file_name,new_img_array)
    # new_img.save(parent_dir+file_name)


    


def inference_xray(path, results_path, jpg_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    image_names = [path]
    unet_model = 'unet'
    chexnet_model = 'chexnet'
    DISEASES = np.array(CLASS_NAMES)

    unet = Unet(trained=True, model_name=unet_model).cpu()
    chexnet = ChexNet(trained=True, model_name=chexnet_model).cpu()
    heatmap_generator = HeatmapGenerator(chexnet, mode='cam')
    unet.eval();
    chexnet.eval()
    print(path)
    PROB_POOL=[]
    INDEX=[]
    print(image_names)
    s_name = []
    for image_name in tqdm(image_names):
        try:
            image = dicom.dcmread(image_name, force=True).pixel_array
            image = image/image.max()
            image= (image*255).astype('uint8')
            xray=image.copy()

            original_size=image.shape

            image = Image.fromarray(image)
            image = image.convert('RGB')
            image = image.resize((1024,1024))
            (t, l, b, r), mask = unet.segment(image)

            cropped_image = image.crop((l, t, r, b))
            prob = chexnet.predict(cropped_image)
            PROB_POOL.append(prob)

            w, h = cropped_image.size

            heatmap, _ = heatmap_generator.from_prob(prob, w, h)


            w, h = cropped_image.size
            heatmap, _ = heatmap_generator.from_prob(prob, w, h)
            p_l, p_t = l, t
            p_r, p_b = 1024-r, 1024-b
            heatmap = np.pad(heatmap, ((p_t, p_b), (p_l, p_r)), mode='linear_ramp', end_values=0)
            heatmap = ((heatmap - heatmap.min()) * (1 / (heatmap.max() - heatmap.min())) * 255).astype(np.uint8)

            heatmap = cv2.resize(heatmap,(original_size[1],original_size[0]))

            binary = threshold(200,255,heatmap.copy())

            contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 


            # idx = np.argsort(-prob)
            # top_prob = prob[idx[:10]]
            top_prob = prob.copy()
            top_prob = map(lambda x: f'{x:.3}', top_prob)

            # top_disease = DISEASES[idx[:10]]
            top_disease = DISEASES.copy()
            prediction = dict(zip(top_disease, top_prob))

            result = {'result': prediction, 'image_name': image_name}

            # df=pd.DataFrame(prediction,index=[0])
            save_name = image_name.split('/')[-1].replace('.dcm','.txt')
            dicom_path = image_name
            multi_data=[]

            try:
                coords = get_1D_coord(contours)
                for coord in coords:
                    data_final = get_json(data = [],coord=coord,input_f = dicom_path)
                    multi_data.append(data_final)
                    
            except Exception as e:
                print(e)
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++',e)
                print('file_name=',image_name)

            json_txt = json.dumps(multi_data)
            f = open(results_path+'/'+save_name,'w')
            f.write(json_txt)
            f.flush()
            s_name.append(results_path+'/'+save_name)
            INDEX.append(image_name)
            save_overlayed_heatmap(heatmap=heatmap,xray=xray,parent_dir=jpg_path,quality_jpeg=50,want_alpha=False,transparency=0.3
                ,threshold_heatmap=True,threshold_min=150)

        except Exception as e:
            print(e)
            print("ERROR FROM OUTER EXCEPT BLOCK")
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++',e)
            print('file_name=',image_name)


    final_df = pd.DataFrame(PROB_POOL,columns=DISEASES,index = INDEX)
    return final_df, s_name
