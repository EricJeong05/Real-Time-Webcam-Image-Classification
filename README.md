# Real-Time-Webcam-Image-Classification
Real-Time Webcam Classifier with CUDA

1. Get a CPU Webcam + Pretrained Classifier pipeline working for baseline
    - Using DETR (Transformer) & ResNET (CNN)
    - Playing with different preprocessing methods (image size, image type, normalization, etc)

    DETR
    - Using PIL images: 6-8 FPS
    - Using OpenCV native pipeline: 8-10 FPS
    - Decreasing image size increases frame rate

    RESNET
    - Using PIL images: 60-65 FPS
    - Using OpenCV native pipeline: 70-80 FPS 
    - Decreasing image size increases frame rate

2. GPU Acceleration with PyTorch
    -  Move the model and input data to GPU memory so inference can be done on the GPU vs CPU

    DETR
    - Avg: 60 FPS

    RESNET
    - Avg: 300+ FPS

3. Real-Time Image Filters with CUDA
    - Create external CUDA kernels and load them into webcam pipeline for real-time image filters
    - Sobel edge detection

4. Combine Filters + Classification
    - 

5. Optimizations (Stretch Goals)
    - 