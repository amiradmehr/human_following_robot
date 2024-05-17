import cv2
import torch
import yaml
import numpy as np
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf,get_iou

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


class Human_Detection:
    def __init__(self,pred_conf=0.85) -> None:
        """Class to handle Human detection and segmentation
        """
        self.features = np.zeros((256,1))
        self.detected = False
        self.zipfile = None
        self.conf_box = []
        self.pred_conf = pred_conf
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            weigths = torch.load('/home/amir/Documents/Robotics Project/data/yolov7-mask.pt')
            with open('/home/amir/Documents/Robotics Project/data/hyp.scratch.mask.yaml','r') as f:
                self.hyp = yaml.load(f, Loader=yaml.FullLoader)   
        except (FileNotFoundError, OSError) as e:
            print(f"Could not load file: {e}")
        else:
            self.model = weigths['model']
            self.model = self.model.to(self.device).float()
            _ = self.model.eval()
            self.person_class_idx = self.model.names.index('person')

    def isdetected(self):
        return self.detected
    
    def configuration(self):
        if(not self.isdetected()):
            return None,None
        
        best_iou = 0.1
        best_box,best_mask = None, None
        for one_mask, bbox, cls, conf in self.zipfile:
            if conf < self.pred_conf or cls != self.person_class_idx:
                continue
            iou = get_iou(self.conf_box,bbox)
            if(iou>best_iou):
                best_iou = iou
                best_box,best_mask = bbox,one_mask
        return best_box,best_mask

        
        

    def detect(self,image):
        image_tensor = self.Preprocessing(image)
        model_output = self.model(image_tensor)
        self.Postprocessing(image_tensor,model_output)

    
    def Preprocessing(self,image):
        """Converts the RGB images to tensor of shape [1, 3, 640, 448]

        Args:
            image (numpy.ndarray): RGB image

        Returns:
            torch.Tensor: tensor of shape [1, 3, 640, 448]
        """
        image_tensor = transforms.ToTensor()(image)
        image_tensor = torch.tensor(np.array([image_tensor.numpy()]))
        image_tensor = image_tensor.to(self.device)
        image_tensor = image_tensor
        return image_tensor
    
    def Postprocessing(self,image_tensor,model_output):
        """_summary_

        Args:
            image_tensor (_type_): _description_
            model_output (_type_): _description_

        Returns:
            _type_: _description_
        """
        _,_,height,width = image_tensor.shape
        inf_out, attn, bases, sem_output = model_output['test'], model_output['attn'], model_output['bases'], model_output['sem']
        bases = torch.cat([bases, sem_output], dim=1)
        pooler_scale = self.model.pooler_scale
        pooler = ROIPooler(output_size=self.hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
        output, output_mask, _, _, _ = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, self.hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
        pred, pred_masks = output[0], output_mask[0]
        if(pred!=None or pred_masks!=None):
            self.detected = True
            bboxes = Boxes(pred[:, :4])
            original_pred_masks = pred_masks.view(-1, self.hyp['mask_resolution'], self.hyp['mask_resolution'])
            pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
            pred_masks_np = pred_masks.detach().cpu().numpy()
            pred_cls = pred[:, 5].detach().cpu().numpy()
            pred_conf = pred[:, 4].detach().cpu().numpy()
            nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int64)
            self.zipfile = list(zip(pred_masks_np, nbboxes, pred_cls, pred_conf))
        else:
            self.detected = False


    
    
    def mask_bg(self,image,bbox,one_mask,color=(0,255,0)):
        img_croppped = image[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        mask = (one_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]).astype(np.uint8)*255
        foreground = cv2.bitwise_or(img_croppped, img_croppped, mask=mask)
                
        mask = cv2.bitwise_not(mask)
        color_background = np.zeros_like(img_croppped, dtype=np.uint8)
        color_background[:] = color
        background = cv2.bitwise_or(color_background, color_background, mask=mask)

        img_masked = cv2.bitwise_or(foreground, background)

        return img_masked
        
        
    def draw(self,image,bbox=None,mask=None,color=(0,0,0)):  
        if mask is not None and mask.any():         
            image[mask] = image[mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        
        if bbox is not None and len(bbox) >= 4:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2) 


##Yet to implement##
# Zipfile cleaning
# Eliminate conf_box initialisation everytime
#mask_bg bug return None type

def main():
    image = cv2.imread('data/person.jpg')
    H_Detect = Human_Detection() 
    img,zipfile = H_Detect.detect(image)
    pnimg = img.copy()

    person_class_idx = H_Detect.model.names.index('person')
    for one_mask, bbox, cls, conf in zipfile:
        if conf < H_Detect.pred_conf and cls !=person_class_idx:
            continue
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]               
        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    cv2.imshow("RESULT",pnimg)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()