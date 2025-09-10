# Real Time Webcam Image Classification with CUDA
This project is mainly for me to get my feet wet with working with pretrained models, running inference, GPU acceleration, and more!
Come follow my ML learning journey!

## 1. Get a CPU Webcam + Pretrained Classifier pipeline working for baseline
**First to get a sense of how to use PyTorch and figure out how to use OpenCV and run models on it, I ran everything on the CPU and just got a simple webcam + model pipeline working as proof of concept**

- I started by using the ResNET18 model, which is a deep convolutional neural network that's lightweight and trained on a set of 1000 classes
(***Important Note: ResNET18 expects a input image size of 224x224***)

- So, images needed to be preprocessed before feeding it into ResNet: resize image to (224x224), convert to expected tensor and channel order, normalize using pretrained ImageNet mean and std

### Baseline CPU performance results:
ResNET: 60-65 FPS (Using PIL images) | 70-80 FPS (Using OpenCV native pipeline)

- The limitation for ResNET, while it was fast and lightweight, was that it only identifies the one major object in the frame and classifies that. I wanted a something that could do object detection and identify the bounded objects like in autonomous driving systems shown below:

![AV object detection example](/images/av_object_detection_example.png)

- So I tried using the DETR (Detection Transformer) model to perform object detection. DETR uses 4 main components:
1. CNN - Similar to ResNET that extracts a 2D feature map from the image
2. Encoder - The feature map is then flattened into a 1D sequence of feature vectors (aka embeddings) along with positional encodings which are passed into the encoder for multi-headed self-attention
3. Decoder - Takes in the learned positional embeddings to reason about the encoded image features.
4. FFN - The feed forward network, a classical MLP, that predicts the final bounding box coordinates and class labels

- While this achieved the style of webcam classifier I wanted, it was very slow and was only trained to classify the 80 labels in the COCO dataset

### Baseline CPU performance results:
DETR: 6-8 FPS (Using PIL images) | 8-10 FPS (Using OpenCV native pipeline)

- Also I switched from converting my openCV input image to PIL to using the openCV native pipeline since that provided slight performance benefits

## 2. GPU Acceleration with PyTorch
- Moved the model inference step and input data to GPU memory so inference can be done on the GPU instead to see if this simple step can improve performance and it did!

### GPU performance results:
DETR: Avg 60 FPS | RESNET: Avg 300+ FPS

## 3. Combined DETR + ResNet Models
- After playing with both models, I thought why not combine the two so that I can get fine grain classification from ResNET + object detection with bounding boxes from DETR? This way I can increase the number of classifiable objects from 80 -> 1000 from the objects detected by DETR.
- So what I now do is run DETR on the input frame first, then for all the objects detected, crop the image within the bounding box and run ResNET on that cropped image. I then display the ResNET classification on top of the DETR bounding box

- The original input image I'm getting from the webcam is 640 x 480 so I didn't do any resizing for my DETR input. I want to keep the image resolution as high as possible so that after cropping out the bounding boxes, I have enough data to reasonabily resize to the expected ResNet input of 224 x 224.

- This definitely had a big hit in performance, dropping the fps around half compared to DETR only
### Performance results:
Avg: 30-40 FPS

## 4. Real-Time Image Filters with CUDA
- Create a guided filter in CUDA to preserve the edge structure while removing noise with faster compute time and load them into webcam pipeline for real-time image filters
- Done to try and improve preprocessing for better overall inference and to learn more about CUDA!

## 5. Combine Filters + Classification
- Run CUDA filter & inference on different CUDA streams in parallel to learn how to use CUDA streams
- No noticeable runtime fps improvements since DETR is still the main big bottleneck 

### Performance results with CUDA filters & CUDA streams:
Avg: 18-20 FPS

# Improving runtime performance!
## 6. Optimizations
- Ran Nsight Systems (nsys) to see where the bottlenecks are. From a initial pass, there was still a big chunk of processing time held up in the CPU space and time was eaten by CPU-GPU memory transfers. So I decided to tackle those areas and try and move as much preprocessing to the GPU and batch all the image crops done by DETR into one to perform a single ResNet forward pass instead of multiple.

- I used pin_memory and to(device) to run normalization and conversion to tensor on GPU before the pass through DETR and I used torch.stack to batch all crops in a single ResNet forward pass, reducing the number of GPU operations

### Final Performance Results & Summary:
When running with a cuda image filter: Avg 25-30fps | Without cuda image filter: Avg 50-55fps

- Doing these optimization steps provided pretty decent gains in FPS on inference: +~44% increase with CUDA image filter running and a +~50% increase when running without a CUDA image filter!
- 
-  Since DETR a heavy model, we take into account that that's not modifiable and I want to just see how fast I can make it with DETR.