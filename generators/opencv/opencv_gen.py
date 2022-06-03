import cv2
import sys
import argparse
import numpy as np
import os
import json



def gen_rain(input_path, output_path, folder=2):
    # if not os.path.exists(os.path.join(output_path, 'source_datasets')):
    #     os.makedirs(os.path.join(output_path, 'source_datasets'))

    # if not os.path.exists(os.path.join(output_path, 'follow_up_datasets')):
    #     os.makedirs(os.path.join(output_path, 'follow_up_datasets'))
        
    source_path = input_path
    img_list = os.listdir(source_path)
    for img_name in img_list:
        if '.png' in img_name:
            img = cv2.imread(os.path.join(source_path, img_name))
            # cv2.imwrite(os.path.join(output_path, 'x_n1', img_name), img)
            noise = get_noise(img,value=200)
            rain = rain_blur(noise,length=30,angle=-30,w=3)
            rain_img = alpha_rain(rain,img,beta=0.6)  #方法一，透明度賦值
            cv2.imwrite(os.path.join(output_path, 'x_n{}'.format(folder), img_name), rain_img)

def get_noise(img,value=10):  
    
    noise = np.random.uniform(0,256,img.shape[0:2])

    v = value *0.01
    noise[np.where(noise<(256-v))]=0
    
    

    k = np.array([ [0, 0.1, 0],
                    [0.1,  8, 0.1],
                    [0, 0.1, 0] ])
            
    noise = cv2.filter2D(noise,-1,k)
    

    '''cv2.imshow('img',noise)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    return noise

def rain_blur(noise, length=10, angle=0,w=1):

    trans = cv2.getRotationMatrix2D((length/2, length/2), angle-45, 1-length/100.0)  
    dig = np.diag(np.ones(length))   
    k = cv2.warpAffine(dig, trans, (length, length))  
    k = cv2.GaussianBlur(k,(w,w),0)    

    blurred = cv2.filter2D(noise, -1, k)    
    
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    '''
    cv2.imshow('img',blurred)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    
    return blurred

def alpha_rain(rain,img,beta = 0.8):
    
    rain = np.expand_dims(rain,2)
    rain_effect = np.concatenate((img,rain),axis=2)  #add alpha channel

    rain_result = img.copy()   
    rain = np.array(rain,dtype=np.float32)     
    rain_result[:,:,0]= rain_result[:,:,0] * (255-rain[:,:,0])/255.0 + beta*rain[:,:,0]
    rain_result[:,:,1] = rain_result[:,:,1] * (255-rain[:,:,0])/255 + beta*rain[:,:,0] 
    rain_result[:,:,2] = rain_result[:,:,2] * (255-rain[:,:,0])/255 + beta*rain[:,:,0]

    # cv2.imshow('rain_effct_result',rain_result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return rain_result

def add_rain(rain,img,alpha=0.9):

    #chage rain into  3-dimenis

    rain = np.expand_dims(rain,2)
    rain = np.repeat(rain,3,2)

    result = cv2.addWeighted(img,alpha,rain,1-alpha,1)
    cv2.imshow('rain_effct',result)
    cv2.waitKey()
    cv2.destroyWindow('rain_effct')


def hex2rgb(hex):
   hex = hex.lstrip('#')
   rgb =  [int(hex[i:i+2], 16) for i in (0, 2, 4)]
   return rgb[::-1]

nonaddable_hex = {
    "#ff0000": "Car 1",
    "#c80000": "Car 2",
    "#960000": "Car 3",
    "#800000": "Car 4",
    "#b65906": "Bicycle 1",
    "#963204": "Bicycle 2",
    "#5a1e01": "Bicycle 3",
    "#5a1e1e": "Bicycle 4",
    "#cc99ff": "Pedestrian 1",
    "#bd499b": "Pedestrian 2",
    "#ef59bf": "Pedestrian 3",
    "#ff8000": "Truck 1",
    "#c88000": "Truck 2",
    "#968000": "Truck 3",
    "#000064": "Tractor",
    "#b432b4": "Drivable cobblestone",
    "#00ff00": "Small vehicles 1",
    "#00c800": "Small vehicles 2",
    "#009600": "Small vehicles 3",
    "#0080ff": "Traffic signal 1",
    "#1e1c9e": "Traffic signal 2",
    "#3c1c64": "Traffic signal 3",
    "#00ffff": "Traffic sign 1",
    "#1edcdc": "Traffic sign 2",
    "#3c9dc7": "Traffic sign 3",
    "#ffff00": "Utility vehicle 1",
    "#ffffc8": "Utility vehicle 2",
    "#c87dd2": "Painted driv. instr.",
    "#eee9bf": "Slow drive area",
    "#8000ff": "Dashed line",
    "#eea2ad": "Grid structure",
    "#960096": "RD restricted area",
    "#ffc125": "Solid line",
    "#f1e6ff": "Buildings",
    "#ff0080": "Obstacles / trash",
    "#fff68f": "Poles",
    "#400040": "Irrelevant signs",
    "#ff00ff": "RD normal street",
}
nonaddable = [hex2rgb(h) for h in nonaddable_hex.keys()]

def filter(area):
    # for i in range(area.shape[0]):
    #     for j in range(area.shape[1]):
            # print(area[i, j])
    # check_area = list(area[-10:, 0]) + list(area[-1,-10:])
    check_area = [area[-1, 0], area[-1, -1], area[100, 0], area[100, -1]]
    for p in check_area:
        if list(p) in nonaddable:
            return True
    return False


def gen_data_single(source_image, mask):
    img = cv2.imread(source_image)
    mask2 = np.zeros((mask.shape[0], mask.shape[1], 1))
    mask2 = mask[:, :, 0:1] + mask[:, :, 1:2] + mask[:, :, 2:]
    mask2[np.nonzero(mask2)] = 1
    img = img - img*mask2
    img = img + mask
    img = img[161:1208, 442:1489]
    resize_img = cv2.resize(img, (224, 224))     
    return resize_img


def scale(mask, ratio, shift_up=0, shift_left=0):
    new_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    mask_position = np.nonzero(mask)
    height_up = mask_position[0].min()
    height_down = mask_position[0].max()
    width_left = mask_position[1].min()
    width_right = mask_position[1].max()
    mask_object = mask[height_up:height_down, width_left:width_right]
    mask_object_scale = cv2.resize(mask_object, (int(ratio*mask_object.shape[1]), int(ratio*mask_object.shape[0])))

    new_mask[height_up+shift_up:height_up+mask_object_scale.shape[0]+shift_up, width_left+shift_left:width_left+mask_object_scale.shape[1]+shift_left] = mask_object_scale
    return new_mask

#b496c8
# (250, 150, 180)
def gen_data(mask_file, dataset_path, semantic_path, output_path, location, phase=1, img_size=(320, 160)):
    # root_path = "E:\\a2d2\\camera_lidar_semantic\\%s\\camera\\cam_front_center" % dataset_path
    mask = cv2.imread(mask_file)
    # mask = np.concatenate([mask, np.zeros((1208, 150, 3))],axis=1)
    # mask = mask[:, 150:, :]

    # mask = cv2.cvtColor(mask, cv2.cvtColor)
    i = 0
    image_list = os.listdir(dataset_path)
    image_list.sort()

    if "closer" in location:
        op_mask = scale(mask, 1.5, shift_up=50)
        # op_mask = mask
    else:
        op_mask = np.copy(mask)

    for d in image_list:
        if i < len(image_list):
            i += 1
            if 'png' in d:
                img = cv2.imread(os.path.join(dataset_path, d))
                semantic_img = cv2.imread(os.path.join(semantic_path, d.replace('camera', 'label')))



                mask_position = np.nonzero(op_mask)
                height_up = mask_position[0].min()
                height_down = mask_position[0].max()
                width_left = mask_position[1].min()
                width_right = mask_position[1].max()


                mask_object = np.zeros((op_mask.shape[0], op_mask.shape[1], 3))
                added = False

                if "roadside" in location:
                    for j in range(width_left, op_mask.shape[1]):
                        if (semantic_img[height_down, j] == [255, 0, 255]).all():
                            continue
                        try:
                            # add filter
                            if filter(semantic_img[height_up:height_down, j:j+width_right-width_left]):
                                continue

                            mask_object[height_up:height_down, j:j+width_right-width_left] = op_mask[height_up:height_down,
                                                                                             width_left:width_right]
                            mask2 = np.zeros((op_mask.shape[0], op_mask.shape[1], 1))
                            mask2 = mask_object[:, :, 0:1] + mask_object[:, :, 1:2] + mask_object[:, :, 2:]
                            mask2[np.nonzero(mask2)] = 1

                            transform_img = img - img*mask2
                            transform_img = transform_img + mask_object

                            # cv2.rectangle(mask_object, (j, height_up), (j+width_right-width_left+1, height_down+1), (0, 255, 0), 1)
                            # transform_img_with_box = img - img*mask2
                            # transform_img_with_box = transform_img_with_box + mask_object

                        except:
                            break
                        added = True
                        break
                if not added:
                    # print("Image filtered")
                    continue
                if phase == 1:
                    if (j - width_left) > 480:
                        continue
                else:
                    if (j - width_left) > 640:
                        continue
                # transform_img = transform_img[161:1208, 442:1489]
                transform_img = transform_img[248:, :]
                # resize_img = cv2.resize(img[161:1208, 442:1489], img_size)
                resize_img = cv2.resize(img[248:, :], img_size)
                # resize_img = img
                resize_transform_img = cv2.resize(transform_img, img_size)
                # resize_transform_img = transform_img
                # transform_img_with_box = transform_img_with_box[248:, :]
                # resize_transform_img_with_box = cv2.resize(transform_img_with_box, img_size)
                resize_transform_img_with_box = cv2.rectangle(resize_transform_img.copy(), (j//6, (height_up-248) //6),
                                                              ((j+width_right-width_left)//6,
                                                              (height_down-248)//6), (0, 255, 0), 1)
                image_json = d[:-4] + '.json'

                with open(os.path.join(dataset_path,image_json), 'r') as f:
                    image_info = json.load(f)
                    timestamp = image_info["cam_tstamp"]
                    # cv2.imwrite(os.path.join(root_path, "camera_resize", folder, str(timestamp) + '.png'), resize_img)

                    if phase == 1:
                        cv2.imwrite(os.path.join(output_path, 'x_n1', str(timestamp) + '.png'), resize_img)

                    cv2.imwrite(os.path.join(output_path, 'x_n' + str(phase+1), str(timestamp) + '.png'),
                                resize_transform_img)
                    # cv2.imwrite(os.path.join(output_path, 'x_n' + str(phase+2), str(timestamp) + '.png'),
                    #             transform_img)
                    # print(str(timestamp) + '.png', j - width_left)

                    if not os.path.exists(os.path.join(output_path, 'x_n' + str(phase+1) + '_bounding_box')):
                        os.mkdir(os.path.join(output_path, 'x_n' + str(phase+1) + '_bounding_box'))

                    cv2.imwrite(os.path.join(output_path, 'x_n' + str(phase+1) + '_bounding_box', str(timestamp) + '.png'),
                                resize_transform_img_with_box)
                  
                    # print(os.path.join(output_path, str(timestamp) + '.png'))
        else:
            break


def image_control(object, transformation, parameter,  dataset_path, semantic_path, output_path, phase=1):
    mask = None
    if "pedestrian" in object:
        mask = "generators/opencv/person.png"
    elif "sign" in object:
        mask = "generators/opencv/speed_sign.png"

    gen_data(mask, dataset_path, semantic_path, output_path, parameter, phase=phase)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mask", type=str)
    parser.add_argument("--object", type=str)
    parser.add_argument("--location", type=int)
    # parser.add_argument("--object", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_path", type=str)
    # parser.add_argument("--x_n", type=str)
    # parser.add_argument("--img_size", type=int, default=224)

    args = parser.parse_args()
    mask = None
    if args.object == "pedestrian":
        mask = "generators/OpenCV/person.png"
    elif args.object == "bicycle":
        mask = "../generators/OpenCV/bicycle_50.png"
    elif args.object == "vehicle":
        if args.location == 50:
            mask = "../generators/OpenCV/car_50.png"
        elif args.location == 30:
            mask = "../generators/OpenCV/car_30.png"
    elif args.object == "traffic_sign":
        mask = "../generators/OpenCV/speed_30.png"
    # print(mask)
    gen_data(mask, args.location, args.dataset_path,args.output_path, args.img_size)