import numpy as np
import cv2
from ctypes import *
import os
import ctypes

# Load the CUDA library
try:
    dll_path = "../Real-Time-Webcam-Image-Classification/cuda/exports/guided_filter.dll"
    guided_filter = ctypes.windll.LoadLibrary(dll_path)  # Use windll instead of cdll for Windows
    print(f"Successfully loaded {dll_path}")
    
    # Verify the function exists
    if not hasattr(guided_filter, 'ApplyGuidedFilter'):
        print("Error: 'ApplyGuidedFilter' function not found in DLL")
        print("Available functions:", [func for func in dir(guided_filter) if not func.startswith('_')])
        exit(1)
except Exception as e:
    print(f"Error loading DLL: {e}")
    exit(1)

def ApplyGuidedFilter(image, radius=2, epsilon=0.1):
    # Ensure image is in the correct format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    height, width, channels = image.shape
    
    # Prepare output array
    output = np.zeros_like(image)
    
    # Get the function from the DLL
    apply_filter = guided_filter.ApplyGuidedFilter
    apply_filter.argtypes = [
        POINTER(c_ubyte),    # input image
        POINTER(c_ubyte),    # output image
        c_int,               # width
        c_int,               # height
        c_float,             # epsilon
        c_int,               # radius
        c_int                # numChannels
    ]
    
    # Convert numpy arrays to ctypes pointers
    input_ptr = image.ctypes.data_as(POINTER(c_ubyte))
    output_ptr = output.ctypes.data_as(POINTER(c_ubyte))
    
    # Call the CUDA function
    guided_filter.ApplyGuidedFilter(
        input_ptr,
        output_ptr,
        width,
        height,
        epsilon,
        radius,
        channels
    )
    
    return output

if __name__ == "__main__":
    # Read test image
    image = cv2.imread("../Real-Time-Webcam-Image-Classification/scripts/test_image.jpg")
    if image is None:
        print("Error: Could not read image")
        exit(1)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply guided filter with different parameters
    radius_values = [2, 4, 8]
    epsilon_values = [0.1, 0.2, 0.4]
    
    for radius in radius_values:
        for epsilon in epsilon_values:
            # Apply filter
            filtered = ApplyGuidedFilter(image, radius=radius, epsilon=epsilon)
            
            # Convert back to BGR for saving
            filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
            
            # Save result
            output_name = f"../Real-Time-Webcam-Image-Classification/scripts/filtered images/filtered_r{radius}_e{epsilon}.jpg"
            cv2.imwrite(output_name, filtered_bgr)
            print(f"Saved {output_name}")
    
    print("Done!")
