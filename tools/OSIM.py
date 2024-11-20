#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch
import numpy as np

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess, vis
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument(
        "--ref_path", default="/home/user/ssd2/NeRFBaselines/Mip-NeRF360_dataset/bicycle/GT/", help="path to ref images"
    )
    parser.add_argument(
        "--test_path", default="/home/user/ssd2/NeRFBaselines/Mip-NeRF360_dataset/bicycle/3DGS/", help="path to test images"
    )
    parser.add_argument(
        "--model_name", default="3DGS", help="path to test images"
    )
    parser.add_argument(
        "--dataset", default="Mip-NeRF360_dataset", help="path to test images"
    )
    parser.add_argument(
        "--data", default="bicycle", help="path to test images"
    )
    parser.add_argument(
        "--topk", default="5", type=int, help="top confidence scores"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def calc_IoU(true_box, pred_box_list):
    a = true_box
    b = pred_box_list
    b = np.asarray(b)
    a_area = (a[  2] - a[  0]) * (a[  3] - a[  1])
    b_area = (b[:,2] - b[:,0]) * (b[:,3] - b[:,1])
    intersection_xmin = np.maximum(a[0], b[:,0])
    intersection_ymin = np.maximum(a[1], b[:,1])
    intersection_xmax = np.minimum(a[2], b[:,2])
    intersection_ymax = np.minimum(a[3], b[:,3])
    intersection_w = np.maximum(0, intersection_xmax - intersection_xmin)
    intersection_h = np.maximum(0, intersection_ymax - intersection_ymin)
    intersection_area = intersection_w * intersection_h
    union_area = a_area + b_area - intersection_area
    return intersection_area / union_area

def softmax(x, t):
    x = [element / t for element in x]
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def cos_sim(v1, v2):
    v1 = v1.flatten() 
    v2 = v2.flatten() 
    if (np.linalg.norm(v1) == 0 or np.linalg.norm == 0):
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="gpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs, outputs_all = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            if outputs[0] == None:
                return None, None, img_info
        return outputs, outputs_all, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result, model_name, dataset, data):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    outputs_all_list = []
    for image_name in files:
        outputs, outputs_all, img_info = predictor.inference(image_name)
        if outputs:
            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            outputs_all_list.append(outputs_all[0])
        else:
            outputs_all_list.append(torch.zeros(87, device='cuda:0').unsqueeze(0))
            continue
        if save_result:
            save_folder = os.path.join(vis_folder, dataset)
            save_folder = os.path.join(save_folder, data)
            save_folder = os.path.join(save_folder, model_name)
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    return outputs_all_list

def calc_osim(predictor, vis_folder, ref_path, test_path, current_time, save_result, topk, model_name, dataset, data):
    # object detection
    print(dataset, data, model_name)
    ref_result = image_demo(predictor, vis_folder, ref_path, current_time, save_result, "GT", dataset, data) #this is P_{ref} and its index corresponds to i
    test_result = image_demo(predictor, vis_folder, test_path, current_time, save_result, model_name, dataset, data)
    # associate objects using IoU
    asd_test_result = [] # detection results corresponding to each object in each reference image.
    top_conf_list = [[] for _ in range(len(ref_result))] # top-1 class index of confidence for all objects.
    r_i_j_list = [[] for _ in range(len(ref_result))]
    for i in range (len(ref_result)):
        if ref_result[i] == None:
            continue
        new_output = [None for _ in range(ref_result[i].size()[0])]
        for j in range(ref_result[i].size()[0]): # N_{I_i} = ref_result[i].size()[0] : number of the detected objects in image i
            if test_result[i] == None:
                new_output[j] = torch.zeros(test_result[0].size()[1], device='cuda:0').unsqueeze(0)
                continue
            true_box = ref_result[i][j][:4]
            pred_box_list = []
            for a in range(test_result[i].size()[0]):
                pred_box_list.append(test_result[i][a][:4])
            # calc IoU
            pred_box_list = np.array([pred_box_list[idx].cpu() for idx in range(len(pred_box_list))])
            IoU_list = calc_IoU(true_box.cpu(), pred_box_list)
            IoU = max(IoU_list)
            if(IoU > 0.5): 
                new_output[j] = test_result[i][np.argmax(IoU_list)]
            else:
                new_output[j] = torch.zeros(test_result[0].size()[1], device='cuda:0')
        asd_test_result.append(new_output)
    # extract top-k classes
    for i in range(len(ref_result)):
        for j in range(ref_result[i].size()[0]):
            np_ref_result = [ref_result[i][j][idx].cpu() for idx in range(len(ref_result[i][j]))]
            ref_80 = [elem * np_ref_result[4] for elem in np_ref_result[7:]]
            ref_80 = list(enumerate(ref_80))
            ref_80_sorted = sorted(ref_80, key=lambda x: x[1], reverse=True)
            ref_index = [x[0] for x in ref_80_sorted]
            ref_topk = ref_index[:topk] # top-k from ref[i][j]
            correct_class = ref_topk[0]
            top_conf_list[i].append(correct_class)
            np_test_result = [asd_test_result[i][j][idx].cpu() for idx in range(len(asd_test_result[i][j]))]
            test_80 = [elem * np_test_result[4] for elem in np_test_result[7:]]
            test_80 = list(enumerate(test_80))
            test_80_sorted = sorted(test_80, key=lambda x: x[1], reverse=True)
            test_index = [x[0] for x in test_80_sorted]
            test_topk = test_index[:topk] # top-k from test[i][j]
    # union classes
            union_class = list(set(ref_topk) | set(test_topk))
    # normalize
            p_ref = [ref_80[idx][1] for idx in union_class]
            p_ref = softmax(p_ref, 0.3)
            p_test = [test_80[idx][1] for idx in union_class]
            p_test = softmax(p_test, 0.3)
            if (test_80_sorted[j][1] == 0):
                p_test = np.zeros(len(union_class))
                r_i_j = 0
                r_i_j_list[i].append(r_i_j)
    # r_{i, j}
            else:
                r_i_j = cos_sim(p_ref, p_test)
                r_i_j_list[i].append(r_i_j)
    # r_i
        r_i = 0
        s_i = 0
        for j in range(ref_result[i].size()[0]):
            s_i += (ref_result[i][j][2] - ref_result[i][j][0]) * (ref_result[i][j][3] - ref_result[i][j][1])
        for j in range(ref_result[i].size()[0]):
            r_i += r_i_j_list[i][j] * ((ref_result[i][j][2] - ref_result[i][j][0]) * (ref_result[i][j][3] - ref_result[i][j][1])/ s_i)
    # o_l
    exist_objects = []
    for i in range(len(ref_result)):
        for j in range((ref_result[i].size()[0])):
            if top_conf_list[i][j] not in exist_objects:
                exist_objects.append(top_conf_list[i][j])
    o_i_list = []
    obj_area_list = []
    for obj in exist_objects:
        o_l = 0
        obj_area = 0
        count = 0
        for i in range(len(ref_result)):
            for j in range((ref_result[i].size()[0])):
                if top_conf_list[i][j] == obj:
                    o_l += r_i_j_list[i][j]
                    obj_area += (ref_result[i][j][2] - ref_result[i][j][0]) * (ref_result[i][j][3] - ref_result[i][j][1])
                    count += 1
        o_l /= count
        obj_area /= count
        obj_area = float(obj_area.cpu())
        o_i_list.append(o_l)
        obj_area_list.append(obj_area) # average bbox area
    for obj in range(len(exist_objects)):
        print(exist_objects[obj], COCO_CLASSES[exist_objects[obj]], o_i_list[obj])
    # OSIM
    OSIM = 0
    total_area = 0
    for obj in range(len(exist_objects)):
        total_area += obj_area_list[obj]
    for obj in range(len(o_i_list)):
        OSIM += o_i_list[obj] * (obj_area_list[obj] / total_area)
    print("OSIM: ", OSIM)
    return OSIM

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    calc_osim(predictor, vis_folder, args.ref_path, args.test_path, current_time, args.save_result, args.topk, args.model_name, args.dataset, args.data)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
