from collections import OrderedDict
from engine import Engine

import torch
import time
from rmt_prototype import clear_test_images, pipeline, make_prediction_steer
import pandas as pd
import os
from model_a2d2 import EpochSingle
import shutil
from create_compare_image import create_compare_image
from rmt_prototype import load_config



# verification rule 1
def rule_evaluation(rule, test_models, compare_image_folder, gen_image=True):
    f = open('exp_result.txt', 'a')
    f.write('\n')
    f.write('Rule:' + rule + '\n')
    config = load_config('configuration.yaml')
    generator_list = config['engine']
    models = config['model']
    datasets = config['dataset']
    # location_list, object_list, transformation_list, generator_list, model_list, data_list = parse_new()
    # gen_image = True
    # parser = Parser()
    # rule = values["rule"]
    # target_transformation, target_object, target_parameter, MR_relation, work_engines = parser.rule_parse(rule)
    for test_model in test_models:
        print(test_model)
        values = {"rule": rule, "MRV_pair_data": "A2D2", "MRV_MUT": test_model}
        start_time = time.time()
        _, _, violation, total, _ = pipeline(values,
                                    generator_list, models, datasets,
                                    compare_image_folder=compare_image_folder + '_' + test_model, gen_image=gen_image)
        end_time = time.time()
        # print('Rule:' + rule)
        f.write('test model %s, total generated test cases: %d, violation: %d, time cost: %f' %
              (test_model, total, violation, end_time - start_time) + '\n')

        # print('test model %s, total generated test cases: %d, violation: %d' %
        #       (test_model, total, violation))
        gen_image = False
    f.close()

# def verification_rule_eval(gen_image=True):
#     rule_evaluation(rule_1, test_models, 'verification_rule_1', gen_image=gen_image)
    # rule_evaluation(rule_2, test_models, 'verification_rule_2', gen_image=gen_image)


def rule_1_eval(gen_image=True):
    print("Rule 1")
    rule_1 = "If: a pedestrian appears on the roadside\nThen: the ego-vehicle should slow down.\nIf:\nThen:"
    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]
    rule_evaluation(rule_1, test_models, 'verification_rule_1', gen_image=gen_image)

def rule_2_eval(gen_image=True):
    print("Rule 2")
    rule_2 = "If: a speed limit sign appears on the roadside\nThen: the ego-vehicle should slow down.\nIf:\nThen:"
    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]
    rule_evaluation(rule_2, test_models, 'verification_rule_2', gen_image=gen_image)

def rule_7_eval(gen_image=True):
    rule = "If: the driving time changes into night,\nThen: the ego-vehicle should slow down.\nIf:\nThen:"
    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]
    rule_evaluation(rule, test_models, 'validation_rule_1', gen_image=gen_image)

def rule_3_eval():
    rule = "If:a pedestrian appears on the roadside\nThen: the ego-vehicle should slow down at least 30%.\nIf:\nThen:"
    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]
    rule_evaluation(rule, test_models, 'validation_rule_2', gen_image=False)

def rule_4_eval(gen_image=True):
    rule = "If: a pedestrian appears on the roadside,\n Then:  the ego-vehicle should slow down.\n" \
           "If: he gets closer to the ego-vehicle.\n" \
           "Then: the speed should decrease more."
    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]
    rule_evaluation(rule, test_models, 'validation_rule_3', gen_image=gen_image)

def rule_5_eval(gen_image=True):
    rule = "If: lane lines on the road are removed,\nThen: the steering angle of ego-vehicle should keep same.\nIf:" \
           "\nThen:"
    test_models = ["Epoch(steer)", "VGG16(steer)", "Resnet101(steer)"]
    rule_evaluation(rule, test_models, 'validation_rule_4', gen_image=gen_image)

def rule_6_eval(gen_image=True):
    rule = "If: the buildings is replaced with trees,\n" \
           "Then: the steering angle of ego-vehicle should keep same.\nIf:\nThen:"
    test_models = ["Epoch(steer)", "VGG16(steer)", "Resnet101(steer)"]
    rule_evaluation(rule, test_models, 'validation_rule_5', gen_image=gen_image)


    

def comp_exp(method='deeptest_person'):
    models = ['Epoch','VGG16', 'Resnet101']
    for model in models:
        print(model)
        df = pd.read_csv('{}_{}(speed).csv'.format(method, model))
        pred_ori = df['Original pred'].values
        pred_rain = df['add rain'].values
        pred_rain_person = df['add person and rain'].values
        # print(sum(pred_ori >= pred_rain) / len(pred_ori))
        # print(sum(pred_ori >= pred_rain_person) / len(pred_ori))

        violation_1 = 0
        violation_2 = 0
        violation_3 = 0
        for i in range(len(df['Original pred'])):
            if pred_rain[i] >= pred_ori[i]:
                violation_1 += 1
            if pred_rain_person[i] >= pred_ori[i]:
                violation_2 += 1
            if pred_rain[i] >= pred_ori[i] or pred_rain_person[i] >= pred_rain[i]:
                violation_3 += 1
        
        print(violation_1, violation_2, violation_3)
        print(violation_1/len(pred_ori), violation_2/len(pred_ori), violation_3/len(pred_ori))



def deeproad_eval():
    from generators.opencv.opencv_gen import gen_rain, gen_data
    clear_test_images(1)
    clear_test_images(2)
    clear_test_images(3)
    gen_data('generators/opencv/person.png', '/media/yao/新加卷/a2d2/camera_lidar_semantic/20180807_145028/camera/cam_front_center',
    '/media/yao/新加卷/a2d2/camera_lidar_semantic/20180807_145028/label/cam_front_center',
    'test_images', 'roadside', 1)

    from generators.UNIT.unit_gen import day2rain, day2night


    day2rain('test_images/x_n1', 'test_images/x_n3')
    day2rain('test_images/x_n2', 'test_images/x_n4')
    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]

    from model_a2d2 import Vgg16, Resnet101, Epoch
    for model_name in test_models:
        if 'Epoch' in model_name:
            model = Epoch()
            state_dict_parallel = torch.load('models/driving_models/speed/epoch.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)
        elif 'VGG16' in model_name:
            model = Vgg16()
            state_dict_parallel = torch.load('models/driving_models/speed/vgg16.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)   
        else:
            model = Resnet101()
            state_dict_parallel = torch.load('models/driving_models/speed/resnet101.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)       
        
        model = model.cuda()
        model.eval()
        from data_a2d2 import A2D2MT
        from torchvision import transforms
        from rmt_prototype import make_predictions_speed
        root_path = '/media/yao/新加卷/a2d2'

        mt_set_x1 = A2D2MT(root_path, mt_path="test_images/x_n2", get_mt=False,
                               transform=transforms.ToTensor())
        mt_set_x2 = A2D2MT(root_path, mt_path="test_images/x_n2", get_mt=True,
                               transform=transforms.ToTensor())
        pred_x1, pred_x2 = make_predictions_speed(model, mt_set_x1, mt_set_x2)

        mt_set_x3 = A2D2MT(root_path, mt_path="test_images/x_n3", get_mt=True, transform=transforms.ToTensor())
        pred_x1, pred_x3 = make_predictions_speed(model, mt_set_x1, mt_set_x3)

        mt_set_x3 = A2D2MT(root_path, mt_path="test_images/x_n4", get_mt=True, transform=transforms.ToTensor())
        pred_x1, pred_x4 = make_predictions_speed(model, mt_set_x1, mt_set_x3)

        common_image_list = os.listdir('test_images/x_n1')
        common_image_list.sort()
        common_image_list = common_image_list[1:]
        compare_image_folder = 'deeproad_person_{}'.format(model_name)
        df = pd.DataFrame({"Image name": common_image_list, "Original pred": pred_x1, "add person": pred_x2,
                           "add rain": pred_x3, "add person and rain": pred_x4})
        df.to_csv(compare_image_folder + ".csv")


def deeptest_eval(gen_image=True):
    from generators.opencv.opencv_gen import gen_rain, gen_data

    # clear_test_images(1)




    gen_data('generators/opencv/speed_sign.png', '/media/yao/新加卷/a2d2/camera_lidar_semantic/20180807_145028/camera/cam_front_center',
    '/media/yao/新加卷/a2d2/camera_lidar_semantic/20180807_145028/label/cam_front_center',
    'test_images', 'roadside', 1)

    gen_rain('test_images/x_n1', 'test_images', 3)
    gen_rain('test_images/x_n2', 'test_images', 4)


    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]
    # rule_evaluation(rule, test_models, 'complex_rule_3', gen_image=gen_image) 
    # clear_test_images(0)
    # clear_test_images(1)
    # clear_test_images(2)


    # from generators.pix2pixHD import pix2pixHD_gen
    # pix2pixHD_gen.image_control_complex('/media/yao/新加卷/a2d2/camera_lidar_semantic/20180807_145028/camera/cam_front_center', 'test_images')

    from model_a2d2 import Vgg16, Resnet101, Epoch
    for model_name in test_models:
        if 'Epoch' in model_name:
            model = Epoch()
            state_dict_parallel = torch.load('models/driving_models/speed/epoch.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)
        elif 'VGG16' in model_name:
            model = Vgg16()
            state_dict_parallel = torch.load('models/driving_models/speed/vgg16.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)   
        else:
            model = Resnet101()
            state_dict_parallel = torch.load('models/driving_models/speed/resnet101.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)       
        
        model = model.cuda()
        model.eval()
        from data_a2d2 import A2D2MT
        from torchvision import transforms
        from rmt_prototype import make_predictions_speed
        root_path = '/media/yao/新加卷/a2d2'

        mt_set_x1 = A2D2MT(root_path, mt_path="test_images/x_n2", get_mt=False,
                               transform=transforms.ToTensor())
        mt_set_x2 = A2D2MT(root_path, mt_path="test_images/x_n2", get_mt=True,
                               transform=transforms.ToTensor())
        pred_x1, pred_x2 = make_predictions_speed(model, mt_set_x1, mt_set_x2)

        mt_set_x3 = A2D2MT(root_path, mt_path="test_images/x_n3", get_mt=True, transform=transforms.ToTensor())
        pred_x1, pred_x3 = make_predictions_speed(model, mt_set_x1, mt_set_x3)

        mt_set_x3 = A2D2MT(root_path, mt_path="test_images/x_n4", get_mt=True, transform=transforms.ToTensor())
        pred_x1, pred_x4 = make_predictions_speed(model, mt_set_x1, mt_set_x3)

        common_image_list = os.listdir('test_images/x_n1')
        common_image_list.sort()
        common_image_list = common_image_list[1:]
        compare_image_folder = 'deeptest_2_{}'.format(model_name)
        df = pd.DataFrame({"Image name": common_image_list, "Original pred": pred_x1, "add person": pred_x2,
                           "add rain": pred_x3, "add person and rain": pred_x4})
        df.to_csv(compare_image_folder + ".csv")


def comp_rule_2_eval(gen_image=True):
    rule = "If: a slow speed sign is added on roadside\n Then: the speed of ego-vehicle should decrease.\n" \
           "If: the sign is added closer to the ego-vehicle\n" \
           "Then: the speed of ego-vehicle should decrease more."
    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]
    rule_evaluation(rule, test_models, 'complex_rule_2', gen_image=gen_image)    

def comp_rule_3_eval(gen_image=True):
    rule = "If: the buildings is replaced with trees,\n the steering angle of ego-vehicle should keep same.\n" \
           "If: lane lines on the road are removed,\n" \
           "Then: the steering angle of ego-vehicle should keep same."
    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]
    # rule_evaluation(rule, test_models, 'complex_rule_3', gen_image=gen_image) 
    # clear_test_images(0)
    # clear_test_images(1)
    # clear_test_images(2)


    # from generators.pix2pixHD import pix2pixHD_gen
    # pix2pixHD_gen.image_control_complex('/media/yao/新加卷/a2d2/camera_lidar_semantic/20180807_145028/camera/cam_front_center', 'test_images')
    from model_a2d2 import Vgg16Single, build_resnet101, EpochSingle
    for model_name in test_models:
        if 'Epoch' in model_name:
            model = EpochSingle()
            state_dict_parallel = torch.load('models/driving_models/steer/epoch.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)
        elif 'VGG16' in model_name:
            model = Vgg16Single()
            state_dict_parallel = torch.load('models/driving_models/steer/vgg16.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)   
        else:
            model = build_resnet101()
            state_dict_parallel = torch.load('models/driving_models/steer/resnet101.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)       
        
        model = model.cuda()
        model.eval()
        pred_x1, pred_x2, pred_x3 = make_prediction_steer(model, 3)    
        relation_function_1 = lambda x1, x2: eval('abs(x2-x1)<=1.39')
        relation_function_2 = lambda x2, x3: eval('abs(x3-x2)<=1.39')
        total = len(pred_x1)
        violation = 0
        violation_record = []

        for i in range(total):
            if not relation_function_1(pred_x1[i], pred_x2[i]) or not relation_function_2(pred_x2[i], pred_x3[i]):
                violation += 1
                violation_record.append(1)
            else:
                violation_record.append(0)

        print(violation, total)
        common_image_list = os.listdir('test_images/x_n1')
        common_image_list.sort()
        common_image_list = common_image_list[1:]
        compare_image_folder = 'comp_rule_3_{}'.format(model_name)
        df = pd.DataFrame({"Image name": common_image_list, "Original pred": pred_x1, "Transformed pred_1": pred_x2,
                           "Transformed pred_2": pred_x3, })
        df.to_csv(compare_image_folder + ".csv")
        # if os.path.exists(compare_image_folder):
        #     clear_compare_images(compare_image_folder)
        create_compare_image((pred_x1, pred_x2, pred_x3), common_image_list, violation_record, compare_image_folder,
                             'Steer angle', False)            


def comp_rule_4_eval():
    # from generators.opencv.opencv_gen import gen_data
    # gen_data('generators/opencv/person.png', '/media/yao/新加卷/a2d2/camera_lidar_semantic/20180807_145028/camera/cam_front_center',
    # '/media/yao/新加卷/a2d2/camera_lidar_semantic/20180807_145028/label/cam_front_center',
    # 'test_images', 'roadside', 2)
    # from generators.UNIT.unit_gen import day2night
    # day2night('test_images/x_n3', 'test_images/x_n3')

    # pass
    test_models = ["Epoch(speed)", "VGG16(speed)", "Resnet101(speed)"]
    # rule_evaluation(rule, test_models, 'complex_rule_3', gen_image=gen_image) 
    # clear_test_images(0)
    # clear_test_images(1)
    # clear_test_images(2)


    # from generators.pix2pixHD import pix2pixHD_gen
    # pix2pixHD_gen.image_control_complex('/media/yao/新加卷/a2d2/camera_lidar_semantic/20180807_145028/camera/cam_front_center', 'test_images')
    from model_a2d2 import Vgg16, Resnet101, Epoch
    for model_name in test_models:
        if 'Epoch' in model_name:
            model = Epoch()
            state_dict_parallel = torch.load('models/driving_models/speed/epoch.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)
        elif 'VGG16' in model_name:
            model = Vgg16()
            state_dict_parallel = torch.load('models/driving_models/speed/vgg16.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)   
        else:
            model = Resnet101()
            state_dict_parallel = torch.load('models/driving_models/speed/resnet101.pt')
            state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
            model.load_state_dict(state_dict_single)       
        
        model = model.cuda()
        model.eval()
        from data_a2d2 import A2D2MT
        from torchvision import transforms
        from rmt_prototype import make_predictions_speed
        root_path = '/media/yao/新加卷/a2d2'

        mt_set_x1 = A2D2MT(root_path, mt_path="test_images/x_n2", get_mt=False,
                               transform=transforms.ToTensor())
        mt_set_x2 = A2D2MT(root_path, mt_path="test_images/x_n2", get_mt=True,
                               transform=transforms.ToTensor())
        pred_x1, pred_x2 = make_predictions_speed(model, mt_set_x1, mt_set_x2)

        mt_set_x3 = A2D2MT(root_path, mt_path="test_images/x_n3", get_mt=True, transform=transforms.ToTensor())
        pred_x1, pred_x3 = make_predictions_speed(model, mt_set_x1, mt_set_x3)

        # pred_x1, pred_x2, pred_x3 = make_prediction_steer(model, 3)    
        relation_function_1 = lambda x1, x2: eval('x2<x1')
        relation_function_2 = lambda x2, x3: eval('x3<x2')
        total = len(pred_x1)
        violation = 0
        violation_record = []

        for i in range(total):
            if not relation_function_1(pred_x1[i], pred_x2[i]) or not relation_function_2(pred_x2[i], pred_x3[i]):
                violation += 1
                violation_record.append(1)
            else:
                violation_record.append(0)

        print(violation, total)
        common_image_list = os.listdir('test_images/x_n1')
        common_image_list.sort()
        common_image_list = common_image_list[1:]
        compare_image_folder = 'comp_rule_4_{}'.format(model_name)
        df = pd.DataFrame({"Image name": common_image_list, "Original pred": pred_x1, "Transformed pred_1": pred_x2,
                           "Transformed pred_2": pred_x3, })
        df.to_csv(compare_image_folder + ".csv")
        # if os.path.exists(compare_image_folder):
        #     clear_compare_images(compare_image_folder)
        create_compare_image((pred_x1, pred_x2, pred_x3), common_image_list, violation_record, compare_image_folder,
                             'Speed', False)

def create_mturk_csv(folders, transformations):
    # https://rmt-r1.s3.ap-northeast-2.amazonaws.com/validation_rule_1_VGG16(speed)/0_1533646242154970.png
    # url = "http://itseg.org/wp-content/uploads/2021/03/"
    url = "https://rmt-r1.s3.ap-northeast-2.amazonaws.com"
    mturk_file = {"image": [], "transformation": [], "mturk_image":[]}

    for folder, transformation in zip(folders, transformations):
        img_list = os.listdir(folder)
        for img_name in img_list:
            mturk_file["image"].append(url+ '/' + folder.split('/')[-1] + '/' + img_name)
            mturk_file["mturk_image"].append(url + '/mturk/' + folder.split('/')[-1] + img_name)
            mturk_file["transformation"].append(transformation)
            # shutil.copyfile(os.path.join(folder, img_name), os.path.join('test_images/mturk', img_name))

    mturk_csv = pd.DataFrame(mturk_file)
    print(mturk_csv.shape)
    mturk_csv = mturk_csv.sample(n=345, random_state=0)
    mturk_tabs_csv = pd.DataFrame()
    for i in range(1, 24):
        mturk_tabs_csv["image_" + str(i)] = mturk_csv["mturk_image"].values[15*(i-1):15*i]
        mturk_tabs_csv["transformation_" + str(i)] = mturk_csv["transformation"].values[15*(i-1):15*i]

    mturk_csv.to_csv("models/mTurk.csv", index=False)
    mturk_tabs_csv.to_csv("models/mTurk_tabs.csv", index=False)

    for img in mturk_csv["image"].values:
        img_name = img.split('/')[-1]
        img_folder = img.split('/')[-2]
        shutil.copyfile(os.path.join('test_images', img_folder, img_name), os.path.join('test_images/mturk', img_folder + img_name))


def sensitivity_analysis(rule=1, model='Epoch'):
    if rule in [1, 2]:
        filename = 'verification_rule_{}_{}(speed).csv'.format(rule, model)
    elif rule in [5, 6]:
        filename = 'validation_rule_{}_{}(steer).csv'.format(rule - 1, model)
    elif rule == 7:
        filename = 'validation_rule_1_{}(speed).csv'.format( model)

    df = pd.read_csv(filename)
    pred_ori = df['Original pred']
    pred_trans = df['Transformed pred']

    if rule in [1, 2, 7]:
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        thresholds = [0.695, 1.39, 2.083, 2.778, 3.472, 4.167]
    
    vio_th = []
    for threshold in thresholds:
        if rule in [1, 2, 7]:
            # speed_decrease = (pred_ori - pred_trans) / pred_ori
            speed_decrease =  (pred_ori - pred_trans) / (pred_ori + 0.001) 
            violation = speed_decrease <= threshold
        elif rule in [5, 6]:
            steer_deviation = abs(pred_trans - pred_ori)
            violation = steer_deviation > threshold

        df['threshold_{}'.format(threshold)] = violation
        # print(rule, threshold, sum(violation))
        vio_th.append((sum(violation), sum(violation)/len(pred_ori)))
    
    print(rule, model, vio_th, len(pred_ori))


def sensitivity_exp():
    rules = [5]
    models = ['Epoch', 'VGG16', 'Resnet101']
    for rule in rules:
        for model in models:
            sensitivity_analysis(rule=rule, model=model)


def test_driving_model():
    device = torch.device('cuda')
    driving_model = EpochSingle()
    state_dict_parallel = torch.load('models/driving_models/steer/epoch.pt')
    state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
    driving_model.load_state_dict(state_dict_single)
    driving_model = driving_model.to(device)
    driving_model.eval()
    make_prediction_steer(driving_model)

if __name__ == '__main__':
    # comp_exp(method='deeproad_person')
    # deeptest_exp()
    # deeproad_eval()
    # deeptest_eval()
    # run_1_eval(gen_image=True)

    # sensitivity_exp()
    rule_1_eval(gen_image=True)
    # run_2_eval()
    # rule_3_eval()
    # rule_4_eval()
    # rule_5_eval(gen_image=True)
    # rule_6_eval(gen_image=True)
    # comp_rule_2_eval(gen_image=True)
    # comp_rule_3_eval(gen_image=False)
    # comp_rule_4_eval()    
    # rule_7_eval(gen_image=True)
    # folders = ["test_images/verification_rule_1_VGG16(speed)", "test_images/verification_rule_2_VGG16(speed)",
    #            "test_images/validation_rule_1_VGG16(speed)", "test_images/validation_rule_3_VGG16(speed)",
    #            "test_images/validation_rule_4_VGG16(steer)", "test_images/validation_rule_5_VGG16(steer)"]
    
    # transformations = ["Add a person on roadside", "Add a speed slow sign on roadside",
    #                    "Change the driving scene to night", "Add a person on roadside closer to the self-driving vehicle",
    #                    "Remove lane lines on the road", "Replace buildings with trees",]
    
    # create_mturk_csv(folders, transformations)
    # test_driving_model()