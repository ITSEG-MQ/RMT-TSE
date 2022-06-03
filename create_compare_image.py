import cv2
import os
from steer_viewer import draw
import numpy as np


def create_compare_image(preds, image_list, violation_record, folder_name, label_type='Steer angle',
                         bounding_box=False):
    font = cv2.FONT_HERSHEY_COMPLEX
    # image_list = os.listdir('test_images/x_n1')
    if not folder_name:
        folder_name = 'current_rule'
    if not os.path.exists('test_images/' + folder_name):
        os.mkdir('test_images/' + folder_name)
    if len(preds) == 2:
        for i, image_name in enumerate(image_list):
            # if violation_record[i] == 1:
            x_n1 = cv2.imread(os.path.join('test_images', 'x_n1', image_name))
            if bounding_box:
                x_n2 = cv2.imread(os.path.join('test_images', 'x_n2_bounding_box', image_name))
            else:
                x_n2 = cv2.imread(os.path.join('test_images', 'x_n2', image_name))
            if label_type == "Steer angle":
                x_n1 = draw(x_n1, (preds[0][i] / 25))
                x_n2 = draw(x_n2, (preds[1][i] / 25), y_true=(preds[0][i] / 25))

            compare_image = np.zeros((x_n1.shape[0] + 100, x_n1.shape[1] * 2 + 60, 3)).astype("uint8")
            compare_image[:, :, :] = 255
            compare_image[:x_n1.shape[0], :x_n1.shape[1]] = x_n1
            compare_image[:x_n1.shape[0], x_n1.shape[1] + 60:compare_image.shape[1]] = x_n2
            cv2.putText(compare_image, 'Original image', (10, x_n1.shape[0] + 30), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, 'Transformed image', (x_n1.shape[1] + 60, x_n1.shape[0] + 30), font, 0.8, (0, 0, 0)
                        , 1, cv2.LINE_AA)
            if label_type == "Steer angle":
                if preds[0][i] >= 0:
                    cv2.putText(compare_image, label_type + ' prediction: ',
                                (10, x_n1.shape[0] + 60), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(compare_image, 'turn left %.2f degrees' % (preds[0][i]),
                                (10, x_n1.shape[0] + 90), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(compare_image, label_type + ' prediction: ',
                                (10, x_n1.shape[0] + 60), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(compare_image, 'turn right %.2f degrees' % (-preds[0][i]),
                                (10, x_n1.shape[0] + 90), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                if preds[1][i] >= 0:
                    cv2.putText(compare_image, label_type + ' prediction: ',
                                (x_n1.shape[1] + 60, x_n1.shape[0] + 60), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(compare_image, 'turn left %.2f degrees' % (preds[1][i]),
                                (x_n1.shape[1] + 60, x_n1.shape[0] + 90), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(compare_image, label_type + ' prediction: ',
                                (x_n1.shape[1] + 60, x_n1.shape[0] + 60), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(compare_image, 'turn right %.2f degrees' % (-preds[1][i]),
                                (x_n1.shape[1] + 60, x_n1.shape[0] + 90), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(compare_image, label_type + ' prediction: ',
                            (10, x_n1.shape[0] + 60), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(compare_image, '%.2f mph' % (preds[0][i] / 1.609),
                            (10, x_n1.shape[0] + 90), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(compare_image, label_type + ' prediction: ',
                            (x_n1.shape[1] + 60, x_n1.shape[0] + 60), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(compare_image, '%.2f mph' % (preds[1][i] / 1.609),
                            (x_n1.shape[1] + 60, x_n1.shape[0] + 90), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            if violation_record[i] == 1:
                cv2.imwrite('test_images/' + folder_name + '/1_' + image_name, compare_image)
            else:
                cv2.imwrite('test_images/' + folder_name + '/0_' + image_name, compare_image)

    elif len(preds) == 3:
        for i, image_name in enumerate(image_list):

            x_n1 = cv2.imread(os.path.join('test_images', 'x_n1', image_name))
            if bounding_box:
                x_n2 = cv2.imread(os.path.join('test_images', 'x_n2_bounding_box', image_name))
                x_n3 = cv2.imread(os.path.join('test_images', 'x_n3_bounding_box', image_name))
            else:
                x_n2 = cv2.imread(os.path.join('test_images', 'x_n2', image_name))
                x_n3 = cv2.imread(os.path.join('test_images', 'x_n3', image_name))

            compare_image = np.zeros((x_n1.shape[0] + 100, x_n1.shape[1] * 3 + 120, 3)).astype("uint8")
            compare_image[:, :, :] = 255
            compare_image[:x_n1.shape[0], :x_n1.shape[1]] = x_n1
            compare_image[:x_n1.shape[0], x_n1.shape[1] + 60:x_n1.shape[1] * 2 + 60] = x_n2
            compare_image[:x_n1.shape[0], x_n1.shape[1] * 2 + 120:x_n1.shape[1] * 3 + 120] = x_n3

            cv2.putText(compare_image, 'Original image', (10, x_n1.shape[0] + 30), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, 'Transformed image 1', (x_n1.shape[1] + 60, x_n1.shape[0] + 30), font, 0.8,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, 'Transformed image 2', ((x_n1.shape[1] + 60) * 2, x_n1.shape[0] + 30), font, 0.8,
                        (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(compare_image, label_type + ' prediction:', (10, x_n1.shape[0] + 60), font, 0.7,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, '%.2f mph' % (preds[0][i] / 1.609), (10, x_n1.shape[0] + 90), font, 0.7,
                        (0, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(compare_image, label_type + ' prediction:',
                        (x_n1.shape[1] + 60, x_n1.shape[0] + 60),
                        font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, '%.2f mph' % (preds[1][i] / 1.609), (x_n1.shape[1] + 60, x_n1.shape[0] + 90),
                        font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(compare_image, label_type + ' prediction:',
                        ((x_n1.shape[1] + 60) * 2, x_n1.shape[0] + 60),
                        font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, '%.2f mph' % (preds[1][i] / 1.609), ((x_n1.shape[1] + 60) * 2, x_n1.shape[0] + 90),
                        font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            if violation_record[i] == 1:
                cv2.imwrite('test_images/' + folder_name + '/1_' + image_name, compare_image)
            else:
                cv2.imwrite('test_images/' + folder_name + '/0_' + image_name, compare_image)

def create_compare_image_2(preds, image_list, violation_record, folder_name, label_type='Steer angle', bounding_box=False):
    font = cv2.FONT_HERSHEY_PLAIN
    # image_list = os.listdir('test_images/x_n1')
    if not folder_name:
        folder_name = 'current_rule'
    if not os.path.exists('test_images/' + folder_name):
        os.mkdir('test_images/' + folder_name)
    if len(preds) == 2:
        for i, image_name in enumerate(image_list):
            # if violation_record[i] == 1:
            x_n1 = cv2.imread(os.path.join('test_images', 'x_n1', image_name))
            if bounding_box:
                x_n2 = cv2.imread(os.path.join('test_images', 'x_n2_bounding_box', image_name))
            else:
                x_n2 = cv2.imread(os.path.join('test_images', 'x_n2', image_name))
            if label_type == "Steer angle":
                x_n1 = draw(x_n1, (preds[0][i] / 25))
                x_n2 = draw(x_n2, (preds[1][i] / 25), y_true=(preds[0][i] / 25))

            compare_image = np.zeros((x_n1.shape[0] + 80, x_n1.shape[1] * 2 + 60, 3))
            compare_image[:,:,:] = 255
            compare_image[:x_n1.shape[0], :x_n1.shape[1]] = x_n1
            compare_image[:x_n1.shape[0], x_n1.shape[1] + 60:compare_image.shape[1]] = x_n2
            cv2.putText(compare_image, 'Original image', (10, x_n1.shape[0] + 20), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, 'Transformed image', (x_n1.shape[1] + 60, x_n1.shape[0] + 20), font, 1, (0, 0, 0)
                        , 1, cv2.LINE_AA)
            if label_type == "Steer angle":
                if preds[0][i] >= 0:
                    cv2.putText(compare_image, label_type + ' prediction: ',
                                (10, x_n1.shape[0] + 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(compare_image, 'turn left %.2f degrees' % (preds[0][i]),
                                (10, x_n1.shape[0] + 70), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(compare_image, label_type + ' prediction: ',
                                (10, x_n1.shape[0] + 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(compare_image, 'turn right %.2f degrees' % (-preds[0][i]),
                                (10, x_n1.shape[0] + 70), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

                if preds[1][i] >= 0:
                    cv2.putText(compare_image, label_type + ' prediction: ',
                                (x_n1.shape[1] + 60, x_n1.shape[0] + 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(compare_image, 'turn left %.2f degrees' % (preds[1][i]),
                                (x_n1.shape[1] + 60, x_n1.shape[0] + 70), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(compare_image, label_type + ' prediction: ',
                                (x_n1.shape[1] + 60, x_n1.shape[0] + 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(compare_image, 'turn right %.2f degrees' % (-preds[1][i]),
                                (x_n1.shape[1] + 60, x_n1.shape[0] + 70), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(compare_image, label_type + ' prediction: %.2f mph' % (preds[0][i] / 1.609),
                            (10, x_n1.shape[0] + 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(compare_image, label_type + ' prediction: %.2f mph' % (preds[1][i] / 1.609),
                            (x_n1.shape[1] + 60, x_n1.shape[0] + 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            if violation_record[i] == 1:
                cv2.imwrite('test_images/' + folder_name + '/1_' + image_name, compare_image)
            else:
                cv2.imwrite('test_images/' + folder_name + '/0_' + image_name, compare_image)

    elif len(preds) == 3:
        for i, image_name in enumerate(image_list):

            x_n1 = cv2.imread(os.path.join('test_images', 'x_n1', image_name))
            if bounding_box:
                x_n2 = cv2.imread(os.path.join('test_images', 'x_n2_bounding_box', image_name))
                x_n3 = cv2.imread(os.path.join('test_images', 'x_n3_bounding_box', image_name))
            else:
                x_n2 = cv2.imread(os.path.join('test_images', 'x_n2', image_name))
                x_n3 = cv2.imread(os.path.join('test_images', 'x_n3', image_name))

            compare_image = np.zeros((x_n1.shape[0] + 80, x_n1.shape[1] * 3 + 120, 3))
            compare_image[:,:,:] = 255
            compare_image[:x_n1.shape[0], :x_n1.shape[1]] = x_n1
            compare_image[:x_n1.shape[0], x_n1.shape[1] + 60:x_n1.shape[1]*2 + 60] = x_n2
            compare_image[:x_n1.shape[0], x_n1.shape[1]*2 + 120:x_n1.shape[1]*3+120] = x_n3

            cv2.putText(compare_image, 'Original image', (10, x_n1.shape[0] + 20), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, 'Transformed image 1', (x_n1.shape[1] + 60, x_n1.shape[0] + 20), font, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, 'Transformed image 2', ((x_n1.shape[1] + 60)*2, x_n1.shape[0] + 20), font, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(compare_image, label_type + ' prediction: %.2f mph' % (preds[0][i] / 1.609), (10, x_n1.shape[0] + 50), font, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, label_type + ' prediction: %.2f mph' % (preds[1][i] / 1.609), (x_n1.shape[1] + 60, x_n1.shape[0] + 50),
                        font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(compare_image, label_type + ' prediction: %.2f mph' % (preds[1][i] / 1.609), ((x_n1.shape[1] + 60) * 2, x_n1.shape[0] + 50),
                        font, 1, (0, 0, 0), 1, cv2.LINE_AA)

            if violation_record[i] == 1:
                cv2.imwrite('test_images/' + folder_name + '/1_' + image_name, compare_image)
            else:
                cv2.imwrite('test_images/' + folder_name + '/0_' + image_name, compare_image)