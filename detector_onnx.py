import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import cv2
import sys
import onnxruntime

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)


    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: 
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class Detector():

    def __init__(self):
        super(Detector, self).__init__()
        self.img_size = 416
        self.threshold = 0.25
        self.stride = 1
        self.weights='weights/best.onnx'
        self.init_model()
        self.names = ["nine","ten","jack","queen","king","ace"]        

    def init_model(self):
        
        sess = onnxruntime.InferenceSession(self.weights)
        self.input_name = sess.get_inputs()[0].name
        output_names = []
        for i in range(len(sess.get_outputs())):
            print('output shape:', sess.get_outputs()[i].name)
            output_names.append(sess.get_outputs()[i].name)

        self.output_name = sess.get_outputs()[0].name
        print('input name:%s, output name:%s' % (self.input_name, self.output_name))
        input_shape = sess.get_inputs()[0].shape
        print('input_shape:', input_shape)            
        self.m = sess
        

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0  # 图像归一化
        img = np.expand_dims(img, axis=0)
        assert len(img.shape) == 4

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m.run(None, {self.input_name: img})[0]
        pred = torch.from_numpy(pred).float()
        pred = non_max_suppression(pred, self.threshold, 0.4)

        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in self.names:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes
    




def main():  

    det = Detector()
    cap = cv2.VideoCapture('images/card.mp4')
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
            #print(results) 
        
        cv2.imshow("detector",im)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()

