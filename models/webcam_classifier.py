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

# COCO labels with background
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

edge_lib = ctypes.cdll.LoadLibrary("D:/dev/Real-Time-Webcam-Image-Classification/cuda/edge_detect.dll")
edge_lib.run_edge_detect.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # input
    ctypes.POINTER(ctypes.c_ubyte),  # output
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

class LaunchDETRWebcamClassifier():
    def edge_detect_cuda(self, frame_bgr):
        self.h, self.w, self.c = frame_bgr.shape
        self.inp = frame_bgr.astype(np.uint8).ravel()
        self.out = np.empty_like(self.inp)

        self.inp_ptr = self.inp.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self.out_ptr = self.out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        edge_lib.run_edge_detect(self.inp_ptr, self.out_ptr, self.w, self.h, self.c)

        return self.out.reshape(self.h, self.w, self.c)
 
    def LoadDetrModel(self):
        # Load DETR model from Torch Hub
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        
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
        
        self.normalize = transforms.Normalize(
                mean=imageNet_mean, 
                std=imageNet_std)
        
        # Move model to GPU if available
        print(f"Using device: {device}")
        self.model.to(device)
        self.model.eval()

    def RunDetrModel(self):
        # Opens default webcam (0)
        self.cap = cv2.VideoCapture(0)

        # Used for measuring rolling average FPS
        self.fps_queue = deque()
        self.iter_max = 30
        self.iter = 0
        self.avg_fps = 0

        # Main webcam loop to run inference - the process of using a trained model to make predictions on new, unseen data
        while True:
            # Grab a frame from the webcam
            self.ret_code, self.frame_bgr = self.cap.read()

            # Make sure we got a frame
            if not self.ret_code:
                break
            
            self.start_time = time.time()   # start timer

            # Edge Detection using custom CUDA DLL
            #self.frame_bgr = self.edge_detect_cuda(self.frame_bgr)

            # Convert BGR to RGB (OpenCV uses BGR, but PyTorch expects RGB)
            self.frame_rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)

            # --- PIL Processing ---
            #self.input_tensor = self.transform(self.frame_rgb).unsqueeze(0).to(device)

            # --- OpenCV Processing (faster than PIL) --- 
            self.frame_rgb = cv2.resize(self.frame_rgb, (400, 400))
            self.input_tensor = torch.from_numpy(self.frame_rgb).permute(2,0,1).float() / 255.0 
            self.input_tensor = self.normalize(self.input_tensor).unsqueeze(0).to(device)

            # --- Start: GPU Processing ---
            with torch.no_grad():
                # Forward pass the input through the entire model
                # input_tensor is a tensor of shape 1x3x400x400
                # output is a dict with 'pred_logits' and 'pred_boxes'
                self.outputs = self.model(self.input_tensor)

            # DETR outputs: logits + boxes
            self.logits = self.outputs['pred_logits'][0]
            self.boxes = self.outputs['pred_boxes'][0]
            self.h, self.w, self._ = self.frame_bgr.shape
            self.boxes = self.boxes * torch.tensor([self.w, self.h, self.w, self.h]).to(device)

            self.probs = self.outputs['pred_logits'].softmax(-1)[0, :, :-1]  # drop background
            self.scores, self.labels = self.probs.max(-1)
            # --- End: GPU Processing ---

            # Loop over the detections and draw bounding boxes
            for self.score, self.label, self.box in zip(self.scores, self.labels, self.boxes):
                if self.score.item() < conf_threshold:
                    continue

                self.x_center, self.y_center, self.bw, self.bh = self.box
                self.x1, self.y1 = int(self.x_center - self.bw/2), int(self.y_center - self.bh/2)
                self.x2, self.y2 = int(self.x_center + self.bw/2), int(self.y_center + self.bh/2)

                self.class_name = ""
                self.label_idx = self.label.item()  # e.g., output from a detection model
                if 0 <= self.label_idx < len(COCO_INSTANCE_CATEGORY_NAMES):
                    self.class_name = COCO_INSTANCE_CATEGORY_NAMES[self.label_idx]
                else:
                    self.class_name = f"ID_{self.label_idx}"

                self.text = f"{self.class_name} {self.score:.2f}"

                cv2.rectangle(self.frame_bgr, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)
                cv2.putText(self.frame_bgr, self.text, (self.x1, self.y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
             # --- Measure FPS ---
            self.elapsed = time.time() - self.start_time
            self.fps = 1 / self.elapsed if self.elapsed > 0 else 0``
            self.fps_queue.append(self.fps)

            # rolling average over iter_max iterations
            if self.iter < self.iter_max:
                self.iter += 1
                self.avg_fps = sum(self.fps_queue) / self.iter
            else:
                self.fps_queue.popleft()
                self.avg_fps = sum(self.fps_queue) / self.iter 

            # --- Overlay text ---
            # param list goes: image, text, position, font, font_scale, color, thickness
            cv2.putText(self.frame_bgr, f"{self.avg_fps:.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow("DETR Webcam Object Detection", self.frame_bgr)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break

        self.cap.release()
        cv2.destroyAllWindows()

class LaunchGPUWebcamClassifier():
    # Grab the ImageNet class labels
    def GrabResNetClassLabels(self):
        self.url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(self.url, "imagenet_classes.txt")
        with open("imagenet_classes.txt") as f:
            self.idx_to_labels = [line.strip() for line in f.readlines()]

    # Load ResNet18 pretrained model - ResNet18 is a common model for image classification tasks
    def LoadResNetModel(self):
        # Load ResNet18 model with default weights
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()

        # 1 - Used for PIL Image Processing - Transform webcam frames → tensor
        self.preprocess = transforms.Compose([
            # Takes numpy array from OpenCV (uint8, Height x Width x Color Channels) and converts it to PIL Image
            transforms.ToPILImage(),
            # Resizes the image to the input size expected by ResNet18
            transforms.Resize((224, 224)),
            # Converts the PIL Image to a tensor (CxHxW) and normalizes it from [0,255] to [0,1] (float32)
            transforms.ToTensor(),
            # Normalizes the tensor with ImageNet mean and std using z-score normalization
            transforms.Normalize(
                mean=imageNet_mean, 
                std=imageNet_std
            )
        ])

        # 2 - Used for OpenCV direct image processing - Define normalization (works on tensors directly)
        self.normalize = transforms.Normalize(
                mean=imageNet_mean, 
                std=imageNet_std)
        
        # Move model to GPU if available
        print(f"Using device: {device}")
        self.model.to(device)

    # Main loop to run the model inference on webcam frames
    def RunResNetModel(self):
        # Opens default webcam (0)
        self.cap = cv2.VideoCapture(0)

        # Used for measuring rolling average FPS
        self.fps_queue = deque()
        self.iter_max = 30
        self.iter = 0
        self.avg_fps = 0

        # Main webcam loop to run inference - the process of using a trained model to make predictions on new, unseen data
        while True:
            # Grab a frame from the webcam
            self.ret_code, self.frame_bgr = self.cap.read()

            # Make sure we got a frame
            if not self.ret_code:
                break
            
            self.start_time = time.time()   # start timer

            # Convert BGR to RGB (OpenCV uses BGR, but PyTorch expects RGB)
            self.frame_rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)

            # 1 - PIL Processing - Run our preprocessing pipeline above
            #input_tensor = preprocess(frame_bgr)

            # 2 - OpenCV Processing (+10-15fps faster than PIL)
            self.frame_rgb = cv2.resize(self.frame_rgb, (224, 224))

            # Converts the numpy array (HxWxC) to a tensor (CxHxW) and normalizes it from [0,255] to [0,1] (float32)
            # permute changes the order of dimensions from HWC to CHW
            self.input_tensor = torch.from_numpy(self.frame_rgb).permute(2,0,1).float() / 255.0 
            self.input_tensor = self.normalize(self.input_tensor)

            # Add a batch dimension since PyTorch models expect a batch of images -> shape becomes 1×C×H×W
            # We need to add a batch dimension because the model expects input in the shape of (batch_size, channels, height, width)
            # Neural networks typically process multiple images at once (32-128) during training with the help of GPUs.
            # For inference, we can use a batch size of 1
            self.input_batch = self.input_tensor.unsqueeze(0).to(device)

            # All code within this block disables gradient calculation for inference -> reduces memory usage
            with torch.no_grad():
                # Forward pass through the model
                # input_batch is a tensor of shape 1x3x224x224 and output (logits) is a tensor of shape 1x1000 (1000 classes)
                self.output = self.model(self.input_batch)
                # Convert the output to probabilities (0-1) using softmax & remove the batch dimension
                self.probs = torch.nn.functional.softmax(self.output[0], dim=0)
                # Get the top predicted class and its probability. Returns tensors of shape 1
                self.top_prob, self.top_catid = torch.topk(self.probs, 1)

            # Get the label of the top predicted class
            self.label = self.idx_to_labels[self.top_catid]
            # Get the probability of the top predicted class
            self.prob = self.top_prob.item()

            # --- Measure FPS ---
            self.elapsed = time.time() - self.start_time
            self.fps = 1 / self.elapsed if self.elapsed > 0 else 0
            self.fps_queue.append(self.fps)

            # rolling average over iter_max iterations
            if self.iter < self.iter_max:
                self.iter += 1
                self.avg_fps = sum(self.fps_queue) / self.iter
            else:
                self.fps_queue.popleft()
                self.avg_fps = sum(self.fps_queue) / self.iter 

            # --- Overlay text ---
            self.text = f"{self.label} ({self.prob:.2f}) | {self.avg_fps:.1f} FPS"
            # param list goes: image, text, position, font, font_scale, color, thickness
            cv2.putText(self.frame_bgr, self.text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show video
            cv2.imshow("Webcam Classifier", self.frame_bgr)

            # Exit on 'q'
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break

        # release the webcam and destroy all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # resnet = LaunchGPUWebcamClassifier()
    # resnet.GrabResNetClassLabels()
    # resnet.LoadResNetModel()
    # resnet.RunResNetModel()

    detr = LaunchDETRWebcamClassifier()
    detr.LoadDetrModel()
    detr.RunDetrModel()