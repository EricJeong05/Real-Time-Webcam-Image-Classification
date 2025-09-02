import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import urllib.request
import time
from collections import deque
import ctypes
import numpy as np

conf_threshold = 0.7
# device = "cpu" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ImageNet mean and std are calculated from the original training set
imageNet_mean = [0.485, 0.456, 0.406]
imageNet_std = [0.229, 0.224, 0.225]

# Load custom CUDA DLL for edge detection
edge_lib = ctypes.cdll.LoadLibrary("D:/dev/Real-Time-Webcam-Image-Classification/cuda/edge_detect.dll")
edge_lib.run_edge_detect.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # input
    ctypes.POINTER(ctypes.c_ubyte),  # output
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

class CUDAFilters():
    def edge_detect_cuda(self, frame_bgr):
        self.h, self.w, self.c = frame_bgr.shape
        self.inp = frame_bgr.astype(np.uint8).ravel()
        self.out = np.empty_like(self.inp)

        self.inp_ptr = self.inp.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self.out_ptr = self.out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        edge_lib.run_edge_detect(self.inp_ptr, self.out_ptr, self.w, self.h, self.c)

        return self.out.reshape(self.h, self.w, self.c)

class LaunchWebcamClassifier():
    def __init__(self):
        # Load class labels and models
        self.GrabResNetClassLabels()
        self.GrabCOCOClassLabels()
        self.LoadResNetModel()
        self.LoadDetrModel()

        # Load CUDA Filters
        self.CUDAFilters = CUDAFilters()

        # Used for PIL Image Processing - Transform webcam frames → tensor
        self.transform = transforms.Compose([
            # Takes numpy array from OpenCV (uint8, Height x Width x Color Channels) and converts it to PIL Image
            transforms.ToPILImage(),
            # Resizes the image to the input size expected by ResNet18
            transforms.Resize(400),
            # Converts the PIL Image to a tensor (CxHxW) and normalizes it from [0,255] to [0,1] (float32)
            transforms.ToTensor(),
            # Normalizes the tensor with ImageNet mean and std using z-score normalization
            transforms.Normalize(
                mean=imageNet_mean, 
                std=imageNet_std
            )
        ])
        
        # Used for OpenCV direct image processing - Define normalization (works on tensors directly)
        self.normalize = transforms.Normalize(
                mean=imageNet_mean, 
                std=imageNet_std)
    
    def GrabResNetClassLabels(self):
        # Read from local imagenet_classes.txt file in the labels directory
        with open("labels/imagenet_classes.txt") as f:
            self.imagenet_labels = [line.strip() for line in f.readlines()]
    
    def GrabCOCOClassLabels(self):
        # Read from local COCO_labels.txt file in the labels directory
        with open("labels/COCO_labels.txt") as f:
            self.coco_labels = [line.strip() for line in f.readlines()]
     
    def LoadResNetModel(self):
        # Load ResNet18 model with default weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Move model to GPU if available
        print(f"ResNetModel Using device: {device}")
        self.resnet.to(device)
        self.resnet.eval()

    def LoadDetrModel(self):
        # Load DETR model from Torch Hub
        self.detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        
        # Move model to GPU if available
        print(f"DetrModel Using device: {device}")
        self.detr.to(device)
        self.detr.eval()

    def GrabCOCOOutput(self, label) -> str:
        # Grab COCO class name
        self.class_name = ""
        self.label_idx = label.item()  # output from detection model

        if 0 <= self.label_idx < len(self.coco_labels):
            self.class_name = self.coco_labels[self.label_idx]
        else:
            self.class_name = f"ID_{self.label_idx}"
        
        return self.class_name
        
    def RunModel(self):
        # Opens default webcam (0)
        self.cap = cv2.VideoCapture(0)

        # Used for measuring rolling average FPS
        self.fps_queue = deque()
        self.iter_max = 30
        self.iter = 0
        self.avg_fps = 0

        # Main webcam loop to run inference
        while True:
            # Grab a frame from the webcam
            self.ret_code, self.frame = self.cap.read()

            # Make sure we got a frame
            if not self.ret_code:
                break
            
            self.start_time = time.time()   # start timer

            # Edge Detection using custom CUDA DLL
            #self.frame = self.CUDAFilters.edge_detect_cuda(self.frame)

            # Convert BGR to RGB (OpenCV uses BGR, but PyTorch expects RGB)
            self.frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # --- PIL Processing (on GPU) ---
            # self.input_tensor = self.transform(self.frame_rgb).unsqueeze(0).to(device)

            # --- OpenCV Processing (on GPU) --- 
            self.frame_rgb = cv2.resize(self.frame_rgb, (500, 500))
            # Converts the numpy array (HxWxC) to a tensor (CxHxW) and normalizes it from [0,255] to [0,1] (float32)
            # permute changes the order of dimensions from HWC to CHW
            self.input_tensor = torch.from_numpy(self.frame_rgb).permute(2,0,1).float() / 255.0
            # Normalize
            self.input_tensor = self.normalize(self.input_tensor).unsqueeze(0).to(device)

            # Forward pass the input through detr model            
            with torch.no_grad():
                # input_tensor is a tensor of shape 1x3x400x400
                # output is a dict with 'pred_logits' and 'pred_boxes'
                self.detr_outputs = self.detr(self.input_tensor)

            # DETR outputs: logits + boxes
            self.logits = self.detr_outputs['pred_logits'][0]
            self.boxes = self.detr_outputs['pred_boxes'][0]

            # Rescale boxes to original image dimensions (on GPU)
            self.frame_height, self.frame_width, self._ = self.frame.shape
            self.boxes = self.boxes * torch.tensor([self.frame_width, self.frame_height, self.frame_width, self.frame_height]).to(device)

            # Get class probabilities
            self.probs = self.logits.softmax(-1)[:, :-1]  # drop background
            self.scores, self.labels = self.probs.max(-1)

            # Loop over all the detected objects from DETR
            for self.score, self.label, self.box in zip(self.scores, self.labels, self.boxes):
                # Filter out low confidence detections
                if self.score.item() < conf_threshold:
                    continue
                
                # Convert box format from [center_x, center_y, width, height] to [xmin, ymin, xmax, ymax] for cropping
                self.x_center, self.y_center, self.box_width, self.box_height = self.box
                self.xmin, self.ymin = int(self.x_center - self.box_width/2), int(self.y_center - self.box_height/2)
                self.xmax, self.ymax = int(self.x_center + self.box_width/2), int(self.y_center + self.box_height/2)

                # Crop the detected object from the frame
                self.crop_frame_rgb = self.frame_rgb[max(0, self.ymin):min(self.frame_height, self.ymax), max(0, self.xmin):min(self.frame_width, self.xmax)]
                if self.crop_frame_rgb.size == 0: 
                    continue

                # Run ResNet classification on the cropped frame
                self.crop_frame_rgb = cv2.resize(self.crop_frame_rgb, (224, 224))
                self.crop_tensor = torch.from_numpy(self.crop_frame_rgb).permute(2,0,1).float() / 255.0 
                self.crop_tensor = self.normalize(self.crop_tensor).unsqueeze(0).to(device)
                
                # Forward pass the cropped frame input through resnet model
                with torch.no_grad():
                    self.pred = self.resnet(self.crop_tensor)
                    self.imagenet_id = self.pred.argmax(1).item()
                    self.imagenet_label = self.imagenet_labels[self.imagenet_id]

                # Grab ResNet class name
                # self.coco_label = self.GrabCOCOOutput(self.label)
                self.text = f"{self.imagenet_label}"

                # Draw bounding box and associated label
                cv2.rectangle(self.frame, (self.xmin, self.ymin), (self.xmax, self.ymax), (245, 209, 66), 1)
                cv2.putText(self.frame, self.text, (self.xmin, self.ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 209, 66), 1)
        
            # Measure FPS
            self.elapsed = time.time() - self.start_time
            self.fps = 1 / self.elapsed if self.elapsed > 0 else 0
            self.fps_queue.append(self.fps)

            # Calculate rolling FPS average over 30 iterations
            if self.iter < self.iter_max:
                self.iter += 1
                self.avg_fps = sum(self.fps_queue) / self.iter
            else:
                self.fps_queue.popleft()
                self.avg_fps = sum(self.fps_queue) / self.iter 

            # Overlay FPS value
            # param list goes: image, text, position, font, font_scale, color, thickness
            cv2.putText(self.frame, f"{self.avg_fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Overlay Title
            cv2.imshow("DETR + ResNet Webcam Object Detection", self.frame)

            # Press 'q' to quit
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    web = LaunchWebcamClassifier()
    web.RunModel()