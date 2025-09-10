import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
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
edge_filter = ctypes.cdll.LoadLibrary("../Real-Time-Webcam-Image-Classification/cuda/exports/edge_detect.dll")
edge_filter.run_edge_detect.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # input
    ctypes.POINTER(ctypes.c_ubyte),  # output
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

guided_filter = ctypes.cdll.LoadLibrary("../Real-Time-Webcam-Image-Classification/cuda/exports/guided_filter.dll")
guided_filter.applyGuidedFilter.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # input
    ctypes.POINTER(ctypes.c_ubyte),  # output
    ctypes.c_int, ctypes.c_int,      # width, height
    ctypes.c_float,                   # epsilon
    ctypes.c_int,                     # radius
    ctypes.c_int                      # numChannels
]

class CUDAFilters():
    def edge_detect_cuda(self, frame_rgb):
        h, w, c = frame_rgb.shape
        inp = frame_rgb.astype(np.uint8).ravel()
        out = np.empty_like(inp)

        inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        edge_filter.run_edge_detect(inp_ptr, out_ptr, w, h, c)

        return out.reshape(h, w, c)

    def guided_filter_cuda(self, frame_rgb, radius=4, epsilon=0.2):
        h, w, c = frame_rgb.shape
        inp = frame_rgb.astype(np.uint8).ravel()
        out = np.empty_like(inp)

        inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        guided_filter.applyGuidedFilter(inp_ptr, out_ptr, w, h, ctypes.c_float(epsilon), radius, c)

        return out.reshape(h, w, c)

class LaunchWebcamClassifier():
    def __init__(self):
        # Load class labels and models
        self.GrabResNetClassLabels()
        self.GrabCOCOClassLabels()
        self.LoadResNetModel()
        self.LoadDetrModel()

        # Load CUDA Filters
        self.CUDAFilters = CUDAFilters()

        # Used for OpenCV direct image processing - Define normalization (works on tensors directly)
        self.normalize = transforms.Normalize(
                mean=imageNet_mean, 
                std=imageNet_std).to(device)
    
    def GrabResNetClassLabels(self):
        # Read from local imagenet_classes.txt file in the labels directory
        with open("../Real-Time-Webcam-Image-Classification/labels/imagenet_classes.txt") as f:
            self.imagenet_labels = [line.strip() for line in f.readlines()]
    
    def GrabCOCOClassLabels(self):
        # Read from local COCO_labels.txt file in the labels directory
        with open("../Real-Time-Webcam-Image-Classification/labels/COCO_labels.txt") as f:
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

    def SaveCroppedFrame(self, frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"../Real-Time-Webcam-Image-Classification/crops/crop_{timestamp}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def PreprocessDETRInput(self):
        # Edge Detection using custom CUDA DLL - Uncomment to enable
        # self.frame = self.CUDAFilters.edge_detect_cuda(self.frame)

        # Guided Filter using custom CUDA DLL - Uncomment to enable
        # self.frame = self.CUDAFilters.guided_filter_cuda(self.frame, radius=2, epsilon=0.2)

        # Convert BGR to RGB (OpenCV uses BGR, but PyTorch expects RGB)
        self.frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        
        # Running normalization and conversion to tensor on GPU
        # Converts the numpy array (HxWxC) to a tensor (CxHxW) and normalizes it from [0,255] to [0,1] (float32)
        # permute changes the order of dimensions from HWC to CHW
        self.input_tensor = torch.from_numpy(self.frame_rgb).pin_memory().permute(2,0,1).float() / 255.0
        self.input_tensor = self.input_tensor.to(device, non_blocking=True).unsqueeze(0)
        self.input_tensor = self.normalize(self.input_tensor)

    def ResNetClassify(self):
        # Initialize lists to store crops and their metadata
        self.crops = []
        self.valid_boxes = []
        self.valid_scores = []
        self.valid_labels = []

        # Collect all valid crops
        for score, label, box in zip(self.scores, self.labels, self.boxes):
            # Filter out low confidence detections
            if score.item() < conf_threshold:
                continue
            
            # Convert box format from [center_x, center_y, width, height] to [xmin, ymin, xmax, ymax]
            x_center, y_center, box_width, box_height = box
            xmin, ymin = int(x_center - box_width/2), int(y_center - box_height/2)
            xmax, ymax = int(x_center + box_width/2), int(y_center + box_height/2)
            
            # Crop the detected object
            crop = self.frame_rgb[max(0, ymin):min(self.frame_height, ymax), 
                                max(0, xmin):min(self.frame_width, xmax)]
            
            if crop.size == 0:
                continue
                
            # Convert crop to tensor and preprocess
            crop_tensor = torch.from_numpy(crop).pin_memory().to(device, non_blocking=True)
            crop_tensor = crop_tensor.float() / 255.0
            crop_tensor = crop_tensor.permute(2, 0, 1)
            crop_tensor = torch.nn.functional.interpolate(
                crop_tensor.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False,
                antialias=True
            ).squeeze(0)
            
            # Normalize and add batch dimension
            crop_tensor = self.normalize(crop_tensor)

            # Save crops - Uncomment to save crops
            # self.SaveCroppedFrame(crop)
            
            # Append cropped tensor and metadata to lists
            self.crops.append(crop_tensor)
            self.valid_boxes.append((xmin, ymin, xmax, ymax))
            self.valid_scores.append(score.item())
            self.valid_labels.append(label.item())
        
        # If we have valid crops, process them in a batch
        if self.crops:
            # Batch crops
            batch_crops = torch.stack(self.crops)
            
            # Run batch through ResNet
            with torch.no_grad():
                batch_preds = self.resnet(batch_crops)
                batch_ids = batch_preds.argmax(1)
                
            # Process predictions and draw boxes
            for (xmin, ymin, xmax, ymax), imagenet_id in zip(self.valid_boxes, batch_ids):
                # Get class name
                imagenet_label = self.imagenet_labels[imagenet_id.item()]
                
                # Draw bounding box and label
                cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), (245, 209, 66), 1)
                cv2.putText(self.frame, imagenet_label, (xmin, ymin-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 209, 66), 1)
        
    def RunModel(self):
        # Opens default webcam (0)
        self.cap = cv2.VideoCapture(0)

        # Set resolution to 640x480 (max 1280x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Used for measuring rolling average FPS
        self.fps_queue = deque()
        self.iter_max = 30
        self.iter = 0
        self.avg_fps = 0

        # Current and next frame streams for parallel CUDA streams
        stream_preprocess = torch.cuda.Stream()
        stream_inference = torch.cuda.Stream()
        curr_frame_stream = None
        next_frame_stream = None

        # Main webcam loop to run inference
        while True:
            # Grab a frame from the webcam & make sure we got it
            self.ret_code, self.frame = self.cap.read()
            if not self.ret_code:
                break
            
            # Start timer for FPS calculation
            self.start_time = time.time() 

            ### --- Preprocess Frame on a stream--- ###
            if next_frame_stream is not None:
                with torch.cuda.stream(stream_preprocess):
                    self.PreprocessDETRInput()

            ### --- Infer Frame on another stream--- ###
            if curr_frame_stream is not None:
                with torch.cuda.stream(stream_inference):
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

                    # Run ResNet classification on detected objects
                    self.ResNetClassify()

            ### --- Measure FPS --- ###
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

            # Update streams
            next_frame_stream = self.frame
            curr_frame_stream = next_frame_stream

            # Press 'q' to quit
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    web = LaunchWebcamClassifier()
    web.RunModel()