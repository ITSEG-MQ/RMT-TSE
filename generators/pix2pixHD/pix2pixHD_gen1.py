import json
import sys
import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(dir_path)
sys.path.insert(0, dir_path)

# print(cwd, dir_path)
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import copy
import numpy as np
import cv2
import skimage

# from scipy.misc import imsave, imresize
from imageio import imwrite
import random

'''
{0: 'Sidebars', 1: 'Car', 2: 'Bicycle', 3: 'Pedestrian', 4: 'Truck',
 5: 'Small vehicles', 6: 'Traffic signal', 7: 'Traffic sign', 8: 'Utility vehicle',
 9: 'Speed bumper', 10: 'Curbstone', 11: 'Solid line', 12: 'Irrelevant signs',
 13: 'Road blocks', 14: 'Tractor', 15: 'Non-drivable street', 16: 'Zebra crossing',
 17: 'Obstacles / trash', 18: 'Poles', 19: 'RD restricted area', 20: 'Animals',
 21: 'Grid structure', 22: 'Signal corpus', 23: 'Drivable cobblestone', 24: 'Electronic traffic',
 25: 'Slow drive area', 26: 'Nature object', 27: 'Parking area', 28: 'Sidewalk',
 29: 'Ego car', 30: 'Painted driv. instr.', 31: 'Traffic guide obj.', 32: 'Dashed line',
 33: 'RD normal street', 34: 'Sky', 35: 'Buildings', 36: 'Blurred area', 37: 'Rain dirt'}
'''
print(dir_path)
opt = TestOptions().parse(save=False)
opt.name = "label2city_a2d2_512p_feat_2"
# opt.checkpoints_dir = os.path.join(dir_path, 'checkpoints')
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# add instance_feat to control image generation
opt.instance_feat = True
opt.use_encoded_image = True
# opt.use_encoded_image = True
# person = np.load('person.npy')
# person[:, 3] += 100
# np.save('person.npy', person)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

def image_control(object, transformation, parameter, dataset_path, semantic_path, output_path, **kwargs):

    file_name = 0
    for i, data in enumerate(dataset):
        if i < len(dataset):
            img_name = data['path'][0].split('/')[-1]
            with open(os.path.join(dataset_path, img_name.replace('label', 'camera').replace('png', 'json')), 'r') as f:
                image_info = json.load(f)
                timestamp = image_info["cam_tstamp"]
                img_name_by_json = str(timestamp) + '.png'
            #     break
            feat_map, feature_clustered = sample_features(data['inst'])
            generated1, feat_map_origin = model.inference_gen(data['label'], data['inst'], data['image'])
            img_path = data['path']
            # print('process image... %s' % img_path)
            proc_start = time.time()
            new_data = {}
            new_data['inst'] = data['inst'].clone()
            new_data['label'] = data['label'].clone()
            new_data['image'] = data['image'].clone()
            try:
                if transformation == "remove":
                    feat_map, new_data = replace_object(feat_map_origin, new_data)
                    generated2, feat_map = model.inference_gen(new_data['label'], new_data['inst'], new_data['image'],
                                                        feat_map=feat_map)
                elif transformation == "replace":
                    feat_map, new_data = replace_object(feat_map_origin, new_data, semantic_class=[35], replace_class=26)
                    generated2, feat_map = model.inference_gen(new_data['label'], new_data['inst'], new_data['image'],
                                                        feat_map=feat_map)


                # im1 =
                save_image(util.tensor2im(generated1.data[0]),
                           output_path + '/x_n1', img_name_by_json)
                save_image(util.tensor2im(generated2.data[0]),
                           output_path + '/x_n2', img_name_by_json)
                proc_end = time.time()
                # print('Time cost of process an image: ', proc_end - proc_start)

            except Exception as e:
                # pass
                print(e)
        else:
            break
def sample_features(inst): 
    # read precomputed feature clusters 
    cluster_path = os.path.join(opt.checkpoints_dir, opt.name, opt.cluster_path)  
    # features_clustered里存储了cityscape 35个类别里每个instance的特征，因此可以控制generator生成每个instance的样子      
    features_clustered = np.load(cluster_path, encoding='latin1', allow_pickle=True).item()

    # randomly sample from the feature clusters
    inst_np = inst.cpu().numpy().astype(int)                                      
    feat_map = torch.Tensor(inst.size()[0], opt.feat_num, inst.size()[2], inst.size()[3])
    for i in np.unique(inst_np):    
        label = i if i < 1000 else i//1000
        if label in features_clustered:
            feat = features_clustered[label]
            cluster_idx = np.random.randint(0, feat.shape[0]) 
                                        
            idx = (inst == int(i)).nonzero()
            for k in range(opt.feat_num):                                    
                feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
    if opt.data_type==16:
        feat_map = feat_map.half()
    return feat_map, features_clustered

# Cityscapes class: 6: ground; 7: road; 11: building; 24: person; 26: car
# def remove_object(feat_map, data, region, modify_type=7):
#     # cluster_path = os.path.join(opt.checkpoints_dir, opt.name, opt.cluster_path)        
#     # features_clustered = np.load(cluster_path, encoding='latin1', allow_pickle=True).item()

#     source_region = (data['inst'] == modify_type).nonzero()
#     source_feat = feat_map[source_region[0][0], :, source_region[0][2], source_region[0][3]]
    
#     for point in region:
#         data['inst'][point[0], point[1], point[2], point[3]] = modify_type
#         data['label'][point[0], point[1], point[2], point[3]] = modify_type
#         feat_map[point[0], :, point[2], point[3]] = source_feat
    
#     return feat_map, data

def replace_object(feat_map, data, semantic_class=[11, 32], replace_class=33):
    # cluster_path = os.path.join(opt.checkpoints_dir, opt.name, opt.cluster_path)        
    # features_clustered = np.load(cluster_path, encoding='latin1', allow_pickle=True).item()
    replace_start = time.time()
    source_region = (data['label'] == replace_class).nonzero()
    if len(source_region) == 0:
        raise Exception("Image filtered: no object to replace")
    source_feat = feat_map[source_region[0][0], :, source_region[0][2], source_region[0][3]]
    replace_region = []

    for c in semantic_class:
        replace_region.append((data['label'] == c).nonzero())
        condlist = [data['label']!=c, data['label']==c] 
        choicelist_label = [data['label'], replace_class]
        choicelist_inst = [data['inst'], replace_class]
        choicelist_feature_0 = [np.expand_dims(feat_map[0][0].detach().cpu().numpy(), axis=(0,1)), source_feat[0].item()]
        choicelist_feature_1 = [np.expand_dims(feat_map[0][1].detach().cpu().numpy(), axis=(0,1)), source_feat[1].item()]
        choicelist_feature_2 = [np.expand_dims(feat_map[0][2].detach().cpu().numpy(), axis=(0,1)), source_feat[2].item()]

        data['label'] = np.select(condlist, choicelist_label)
        data['inst'] = np.select(condlist, choicelist_inst)
        feat_map_new_0 = np.select(condlist, choicelist_feature_0)
        feat_map_new_1 = np.select(condlist, choicelist_feature_1)
        feat_map_new_2 = np.select(condlist, choicelist_feature_2)
        feat_map[0,0,:,:] = torch.from_numpy(feat_map_new_0)
        feat_map[0,1,:,:] = torch.from_numpy(feat_map_new_1)
        feat_map[0,2,:,:] = torch.from_numpy(feat_map_new_2)
    
    data['label'] = torch.from_numpy(data['label']).cuda()
    data['inst'] = torch.from_numpy(data['inst']).cuda()
    # feat_map = torch.from_numpy(feat_map).cuda()
    
    replace_region = np.vstack(replace_region)
    d = data['label'].shape
    # print(len(replace_region) / (d[2] * d[3]))
    if replace_class == 26:
        if len(replace_region) / (d[2] * d[3]) < 0.05:
            raise Exception("Image filtered: no object to replace")
    elif replace_class == 33:
        if len(replace_region) / (d[2] * d[3]) < 0.01:
            raise Exception("Image filtered: no object to replace")
    # replace_region = (data['label'] in semantic_class).nonzero()


    # point_list = []
    # for point in replace_region:
        # if (point[0], point[2], point[3]) not in point_list:
        # data['inst'][point[0], point[1], point[2], point[3]] = replace_class
        # data['label'][point[0], point[1], point[2], point[3]] = replace_class
            # feat_map[point[0], :, point[2], point[3]] = source_feat
            # point_list.append((point[0], point[2], point[3]))
    
    replace_end = time.time()
    # print("Time cost of replacing objects: ", replace_end - replace_start)
    return feat_map, data

def replace_style(feat_map, feature_clustered, data, replace_class=[33]):
    for c in replace_class:
        # print(replace_class)
        if c in feature_clustered.keys():
            feat = feature_clustered[c]
                # cluster_idx = np.random.randint(0, feat.shape[0]) 
        else:
            continue                                                    
        idx = (data['label'] == c).nonzero()
        if len(idx) == 0:
            continue
        current_feat = feat_map[0, :, idx[0,2], idx[0,3]].detach().cpu().numpy().reshape(1, -1)
        dists = np.sum(np.square(current_feat - feat),axis=1)
        dist_sort = np.argsort(dists)
        replace_feat_num = dist_sort[-1]
        # for p in idx:
        for k in range(opt.feat_num):                                    
            feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[replace_feat_num, k]
            # feat_map[p[0], p[1] + k, p[2], p[3]] = feat[0, k]

    
    return feat_map, data

def add_object(feat_map, data, source_file, source_type=26):
    cluster_path = os.path.join(opt.checkpoints_dir, opt.name, opt.cluster_path)        
    features_clustered = np.load(cluster_path, encoding='latin1', allow_pickle=True).item()

    # get a random feature of adding objects from feature clusters
    source_feats = features_clustered[source_type]
    cluster_idx = np.random.randint(0, source_feats.shape[0])
    car_instances = []
    for i in np.unique(data['inst'].numpy().astype(int)):
        if i >= source_type * 1000 and i < (source_type + 1) * 1000:
            car_instances.append(i)
    
    car_instances.sort()
    if len(car_instances) == 0:
        car_instances.append(0)
    # scales = [0.8, 0.9 ,1, 1.1]
    # level = random.randint(0, 3)
    source_object = resize_object(source_file, 1)
    # source_object = np.load(source_file)
    # source_object = (source_object == 1).nonzero() 
    # source_object = np.array(source_object)
    # source_object[:, 3] += 20
    # source_object[:, 2] += 50
    x_min = source_object[0].min()
    x_max = source_object[0].max()
    y_min = source_object[1].min()
    y_max = source_object[1].max()
    half_object = (x_min + x_max) // 2
    for i in range(half_object, x_max + 1):
        for j in range(y_min, y_max + 1):
            if data['label'][0, 0, i, j] == 26 or data['label'][0, 0, i, j] == 24:
                return False, 0, 0

    for i in range(len(source_object[0])):
        feat_map[0, :, source_object[0][i], source_object[1][i]] = torch.Tensor(source_feats[cluster_idx])
    # feat_map[0, 1, source_object[0], source_object[1]] = torch.Tensor(source_feats[cluster_idx][1])
    # feat_map[0, 2, source_object[0], source_object[1]] = torch.Tensor(source_feats[cluster_idx][2])

    data['label'][0, 0, source_object[0], source_object[1]] = source_type
    data['inst'][0, 0, source_object[0], source_object[1]] = torch.tensor(car_instances[-1] + 1)

    return True, feat_map, data

def resize_object(name, scale):
    region = np.load(name)
    if scale != 1:
        img = np.zeros((512, 1024))
        min_row = np.min(region[:, 2])
        min_col = np.min(region[:, 3])
        region[:, 2] -= min_row
        region[:, 3] -=min_col
        img[region[:, 2], region[:, 3]] = 1
        img = cv2.resize(img, None, fx=scale, fy=scale)
        region = (img == 1).nonzero()
        region = np.array(region)
        if scale >= 1.5:
            scale_x = scale_x - 0.3
        if scale < 1:
            min_col -= 100 * (1 - scale)
        else:
            min_col += 100 * (scale - 1)
            
        min_row = int(min_row * scale)
        # min_col = int(min_col * scale_y)
        region[0] += int(min_row)
        region[1] += int(min_col)
    else:
        region = np.array([region[:, 2].reshape(-1), region[:,3].reshape(-1)])
    return region

def save_image(img, address, name):
    # img = img.cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))
    # img = img[0:426, 192:760]
    # img = imresize(img, (224, 224))
    img = cv2.resize(img, (320, 160))
    cv2.imwrite(os.path.join(address, name), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



if __name__ == "__main__":
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    # add instance_feat to control image generation
    opt.instance_feat = True
    # opt.use_encoded_image = True
    # person = np.load('person.npy')
    # person[:, 3] += 100
    # np.save('person.npy', person)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    # create website
    visualizer = Visualizer(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    model = create_model(opt)
    file_name = 0
    for i, data in enumerate(dataset):
        if i < len(dataset):
        #     break
        # new_data = copy.deepcopy(data)
        # get the region of car instance 26002

            feat_map, feature_clustered = sample_features(data['inst'])
            generated1, feat_map = model.inference_gen(data['label'], data['inst'], data['image'])
            # generated1 = model.inference(data['label'], data['inst'], data['image'])

            # visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
            #                     ('synthesized_image', util.tensor2im(generated1.data[0]))])
            img_path = data['path']
            # print('process image... %s' % img_path)
            # visualizer.save_images(webpage, visuals, img_path)

            # bicycle = (data['inst'] == 33000).nonzero().numpy()
            # bicycle[:, 3] -= 300
            # np.save('bicycle.npy', bicycle)
            # rider = (data['inst'] == 25000).nonzero().numpy()
            # rider[:, 3] -= 300
            # np.save('rider.npy', rider)
            # np_region = car_region.numpy()
            # np.save('person.npy', np_region)
            new_data = {}
            new_data['inst'] = data['inst'].clone()
            new_data['label'] = data['label'].clone()
            new_data['image'] = data['image'].clone()
            # feat_map, new_data = replace_object(feat_map, new_data, 35, 26)
            feat_map, new_data = replace_object(feat_map, feature_clustered, new_data)
            # could_add, feat_map, new_data = add_object(feat_map, new_data, 'bicycle.npy', 33)
            # if could_add:
            #     could_add, feat_map, new_data = add_object(feat_map, new_data, 'rider.npy', 25)
            #     if not could_add:
            #         continue
            # else:
            #     continue
            # could_add, feat_map, new_data = add_object(feat_map, new_data, 'car_vert_same_direction_close_distance.npy')
            # if not could_add:
            #     continue
            generated2,feat_map = model.inference_gen(new_data['label'], new_data['inst'], new_data['image'], feat_map=feat_map)
            
            # save_image(util.tensor2im(generated1.data[0]), 'ori_gen/', str(i) + '.jpg')
            # save_image(util.tensor2im(generated2.data[0]), 'mod_gen/', str(i) + '.jpg')

            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                ('synthesized_image', util.tensor2im(generated1.data[0])),
                                ('Modified_label', util.tensor2label(new_data['label'][0], opt.label_nc)),
                                ('Modified_image', util.tensor2im(generated2.data[0]))
                                ])

            # img_path = [img_path[0][:-4] + 'add_a_car.png']
            visualizer.save_images(webpage, visuals, img_path)
        else:
            break


    webpage.save()