# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse
import os
import cv2
import time
import ops
import torch
import numpy as np
from random import sample
from scipy.spatial.transform import Rotation as Sci_R

import tensorrt as trt
import common

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

TRT_LOGGER = trt.Logger()

list_color = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (0,153,255), (153,0,255)]

min_z = 1200
step = 32
num_iter = 500
thre_dis = 0.010   #m
thre_dis1 = 0.020

f_x, f_y, pp_x, pp_y, width, height = 517.623840, 517.720154, 319.173828, 238.091400, 640, 480
# f_x, f_y, pp_x, pp_y, width, height = 345.421112, 345.368286, 320.968750, 180.520447, 640, 360

camera_matrix = np.array([[f_x, 0, pp_x],
                          [0, f_y, pp_y],
                          [0, 0, 1]])
inv_camera_matrix = np.linalg.inv(camera_matrix)


def get_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return None


class PreProcess:
    def __init__(self) -> None:
        self.input_size = [-1, -1]
        self.img_h = self.img_w = -1

    def letterbox(self, img_bgr):
        self.side = float(self.input_size[0])
        if self.img_h > self.img_w:
            self.scale = self.side / self.img_h
            self.pad = int((self.side - self.scale * self.img_w) / 2)
        else:
            self.scale = self.side / self.img_w
            self.pad = int((self.side - self.scale * self.img_h) / 2)

        img_input = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        img_s = cv2.resize(img_bgr, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        if self.img_h > self.img_w:
            img_input[:, self.pad:self.pad+img_s.shape[1], :] = img_s
        else:
            img_input[self.pad:self.pad+img_s.shape[0], :, :] = img_s
        return img_input


class trtDet(PreProcess):
    def __init__(self, path_trt, confidence_thres=0.5, iou_thres=0.) -> None:
        self.input_size = [256, 256]
        self.img_h = self.img_w = -1
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.bboxes = []
        
        with get_engine(
            path_trt
        ) as self.engine, self.engine.create_execution_context() as self.context:
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def preprocess(self, img_bgr):
        self.img_h, self.img_w, _ = img_bgr.shape
        img_input = self.letterbox(img_bgr)
        img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_rgb_norm = np.array(img_rgb, dtype=np.float32) / 255.0
        img_rgb_norm = img_rgb_norm.transpose(2, 0, 1)[np.newaxis, :]
        return img_rgb_norm
    
    def run_tensorrt(self, img_bgr):
        img_rgb_norm = self.preprocess(img_bgr)
        self.inputs[0].host = img_rgb_norm.copy()
        trt_outputs = []
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        preds = [torch.tensor(o.reshape((1, 5, 1344))) for o in trt_outputs]
        self.postprocess(preds)

    def postprocess(self, preds):
        preds = ops.non_max_suppression(preds,
                                        conf_thres=self.confidence_thres,
                                        iou_thres=self.iou_thres)
        preds = [rst.cpu() for rst in preds]
        self.bboxes = []
        for pred in preds:
            pred[:, :4] = ops.scale_boxes(self.input_size, pred[:, :4], (self.img_h, self.img_w, 3))
            for rst in pred:
                dict_bbox = {"bbox": rst[:4].numpy(), "score": rst[4], "class_id": rst[5]}
                self.bboxes.append(dict_bbox)
        # return results

    def split_lr(self, dict_bboxes):
        bboxes = [b for b in dict_bboxes if b['class_id']==0]
        if len(bboxes) == 0:
            return None
        if len(bboxes) > 1:
            max_score = 0.
            for d_bbox in bboxes:
                if d_bbox['score'] > max_score:
                    max_score = d_bbox['score']
                    bbox = d_bbox['bbox']
        else:
            bbox = bboxes[0]['bbox']

        rect = bbox.copy()
        rect[1] = rect[1] + 0.12 * (bbox[1] - bbox[3])   # t
        # rect[1] = rect[1] + 0.2 * (bbox[0] - bbox[2])   # t
        rect[0] = rect[0] + 0.12 * (bbox[0] - bbox[2])    # l
        rect[2] = rect[2] + 0.12 * (bbox[2] - bbox[0])    # r
        rect[3] = rect[3] + 0.12 * (bbox[3] - bbox[1])    # b
        rect = np.clip(rect, [0, 0, 0, 0], [self.img_w - 1, self.img_h - 1, self.img_w - 1, self.img_h - 1]).astype(np.uint16)
        return rect
    

class RegressionModel:
    """Regression model."""
    def __init__(self, trt_model, image_size=[224, 224]):
        self.input_size = image_size
        with get_engine(
            trt_model
        ) as self.engine, self.engine.create_execution_context() as self.context:
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    def __call__(self, roi_bgr_mask):
        roi_rgb_mask = cv2.cvtColor(roi_bgr_mask, cv2.COLOR_BGR2RGB)
        roi_rgb_mask_rs = cv2.resize(roi_rgb_mask, dsize=self.input_size, fx=0., fy=0.).astype(np.float32)
        img_norm = roi_rgb_mask_rs / 255.
        img_norm -= np.array([0.485, 0.456, 0.406])
        img_norm /= np.array([0.229, 0.224, 0.225])
        img_input = img_norm.transpose((2,0,1))[np.newaxis,:]

        self.inputs[0].host = img_input.copy()

        # Ort inference
        trt_outputs = []
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)[0]
        
        h_roi, w_roi, _ = roi_bgr_mask.shape
        pts = np.array(trt_outputs).reshape((-1, 2))
        pts *= np.array([w_roi, h_roi])

        return pts


class GetBookKpts:
    def __init__(self, path_trt_det, path_trt_pts):
        self.model_det = trtDet(path_trt_det, 0.5, 0.)
        self.model_pts = RegressionModel(path_trt_pts)
        self.rect = None
        self.plane = None
        self.kpts_2d = None
        self.kpts_3d = None

    def calc_2d_cood(self, coor3d):
        coor2d = np.dot(camera_matrix, coor3d) / coor3d[2, :]
        return coor2d[:2, :].T
    
    def calc_3d_cood(self, dep16, coor2d):
        z = dep16[coor2d[:, 1], coor2d[:, 0]].reshape(-1, 1) * 1e-3
        coor2d = np.hstack((coor2d, np.ones_like(z)))
        coor2d *= z
        coor3d = np.dot(inv_camera_matrix, coor2d.T).T
        return coor3d

    def get_pts_2d(self, img_bgr):
        # start_time_det = time.time()
        self.model_det.run_tensorrt(img_bgr)
        rect = self.model_det.split_lr(self.model_det.bboxes)
        if rect is None:
            print("No book detected.")
            return None, None
        # end_time_det = time.time()
        # time_det = end_time_det - start_time_det
        # print("Det time / frame: {:.2f} ms".format(1000 * time_det))
        roi_bgr = img_bgr[rect[1]:rect[3], rect[0]:rect[2], :]
        kpts = self.model_pts(roi_bgr)
        kpts += rect[:2]
        return rect, kpts
    
    def fit_plane(self, points):
        points = np.asarray(points)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        cov = np.dot(centered_points.T, centered_points) / points.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        
        if normal[2] < 0:
            normal = -normal
        
        d = -np.dot(normal, centroid)
        plane_params = np.append(normal, d)
        
        distances = np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)
        error = np.sqrt(np.mean(distances**2))
        # print(plane_params, error)
        return plane_params
    
    def get_pts3d_ondesk(self, pts_2d):
        Z_cen = -self.plane[3] / (self.plane[0] * (pts_2d[:, 0] - pp_x) / f_x + self.plane[1] * (pts_2d[:, 1] - pp_y) / f_y + self.plane[2])
        X_cen = Z_cen * (pts_2d[:, 0] - pp_x) / f_x
        Y_cen = Z_cen * (pts_2d[:, 1] - pp_y) / f_y
        
        pts_3d = np.vstack([X_cen, Y_cen, Z_cen])
        return pts_3d.T
    
    def Run(self, img_bgr, img_dep):
        # self.reproj = self.kpts_2d_in = None
        self.plane = self.kpts_3d = None
        self.rect, self.kpts_2d = self.get_pts_2d(img_bgr)
        if self.rect is None:
            print("No book detected!")
            return [[0.] * 3] * 4
        kpts_src = self.kpts_2d
        kpts_2d_in = np.zeros_like(kpts_src)
        kpts_2d_in[0] = kpts_src[0] * 0.9 + kpts_src[1] * 0.05 + kpts_src[3] * 0.05
        kpts_2d_in[1] = kpts_src[1] * 0.9 + kpts_src[0] * 0.05 + kpts_src[2] * 0.05
        kpts_2d_in[2] = kpts_src[2] * 0.9 + kpts_src[3] * 0.05 + kpts_src[1] * 0.05
        kpts_2d_in[3] = kpts_src[3] * 0.9 + kpts_src[2] * 0.05 + kpts_src[0] * 0.05
        self.kpts_2d_in = kpts_2d_in
        kpts_3d_in = self.calc_3d_cood(img_dep, kpts_2d_in.astype(np.uint16))
        if not np.all((kpts_3d_in[:, 2] > 0) & (kpts_3d_in[:, 2] < min_z / 1000.)):
            print("Illegal depth value!")
            return [[0.] * 3] * 4
        self.plane = self.fit_plane(kpts_3d_in)
        self.kpts_3d = self.get_pts3d_ondesk(kpts_src)
        # self.reproj = self.calc_2d_cood(self.kpts_3d)
        # print("3d:\n", self.kpts_3d)
        l_out_put = np.squeeze(self.kpts_3d).tolist()
        
        return l_out_put

    def draw(self, img_bgr):
        if self.rect is None:
            return img_bgr
        color_rect = (0, 0, 0)
        l_01 = np.linalg.norm(self.kpts_3d[0, :] - self.kpts_3d[1, :])
        l_03 = np.linalg.norm(self.kpts_3d[0, :] - self.kpts_3d[3, :])
        l_12 = np.linalg.norm(self.kpts_3d[1, :] - self.kpts_3d[2, :])
        l_23 = np.linalg.norm(self.kpts_3d[2, :] - self.kpts_3d[3, :])
        print("Length of book:", l_01, l_23, l_03, l_12)
        cv2.rectangle(img_bgr, self.rect[:2], self.rect[2:], color_rect, 2)
        for idx, pt in enumerate(self.kpts_2d):
            cv2.circle(img_bgr, pt.astype(np.uint16), 2, list_color[idx], -1)
        return img_bgr
    

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str("./data/imgs"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0., help="NMS IoU threshold")
    args = parser.parse_args()

    # Build model
    #cBookKpts = GetBookKpts(args.model, args.model.replace(".onnx", ".trt"))
    path_trt1 = "./data/checkpoints/det_book.trt"
    path_trt2 = "./data/checkpoints/pts_book.trt"
    cBookKpts = GetBookKpts(path_trt1, path_trt2)

    total = 0
    list_imgs_bgr = []
    for dir_path, dir_names, file_names in os.walk(args.source):
        for name in file_names:
            if name.endswith(".png") and "rgb" in dir_path:
                if True: # total % 10 == 0:
                    path_bgr = os.path.join(dir_path, name)
                    list_imgs_bgr.append(path_bgr)
                    total += 1

    for path_bgr in list_imgs_bgr:#[::-1]:
        img_bgr = cv2.imread(path_bgr)
        path_dep = path_bgr.replace("rgb", "depth")
        dep16 = cv2.imread(path_dep, -1)
        # Inference
        start_time_est = time.time()
        out_put = cBookKpts.Run(img_bgr, dep16)
        end_time_est = time.time()
        time_est = end_time_est - start_time_est
        print("Est time / frame: {:.2f} ms".format(1000 * time_est))

        img_bgr_rst = cBookKpts.draw(img_bgr)
        
        cv2.imshow("rst", img_bgr_rst)
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            break

