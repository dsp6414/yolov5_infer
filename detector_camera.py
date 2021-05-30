import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device
import cv2
import sys
import argparse

class Detector():

    def __init__(self,opt):
        super(Detector, self).__init__()
        self.img_size = opt.img_size
        self.threshold = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.stride = 1
        self.weights= opt.weights
        self.init_model()
        self.names = ["nine","ten","jack","queen","king","ace"]    
        self.detected_labels = ["nine","ten","jack","queen","king","ace"]   
        

    def init_model(self):

        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()            
        self.m = model


    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()     
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, self.iou_thres)

        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in  self.detected_labels:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes
    




def main(opt):  

    det = Detector(opt)
    cap = cv2.VideoCapture(opt.source)
    fps = int(cap.get(5))
    print('fps:', fps)

    while True:

        _, im = cap.read()
        if im is None:
            break        
        _,results = det.detect(im)      
        for result in results:
            box = result[:4]
            left,top,right,bottom = box[0], box[1], box[2], box[3]
            cv2.rectangle(im, (left, top), (right, bottom), (0, 0, 255), thickness=2) 
            cv2.putText(im,'{0}-{1:.2f}'.format(result[4],float(result[5])),(left, top - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), thickness=1)
        
        cv2.imshow("detector",im)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='.pt path(s)')
    
    parser.add_argument('--source', type=str, default='images/card.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    main(opt)

