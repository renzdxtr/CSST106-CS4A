## CCST106 
Topic 1.1: Introduction to Computer Vision and Image Processing

### Machine Problem No. 1: Exploring the Role of Computer Vision and Image Processing in AI
**Introduction to Computer Vision and Image Processing**

Computer Vision is a field of AI that enables systems to interpret and process visual data from the world. It involves techniques to acquire, process, analyze, and understand images to produce numerical or significant information.

Image processing is crucial in AI as it enhances, manipulates, and analyzes images to extract meaningful information. This process is essential for improving the quality of images and making them suitable for further analysis by AI systems.

---

### Types of Image Processing Techniques

1. **Edge Detection:**  
   Edge detection identifies the boundaries within images, allowing AI systems to recognize objects and their shapes, which is vital for tasks such as object detection and image recognition (Kundu, 2024)

2. **Resizing:**  
   Resizing involves altering the dimensions of an image (e.g., either by enlarging or shrinking it) to ensure consistency and efficiency. It helps in normalizing the data and reducing computational load.

3. **Grayscaling:**  
   Grayscaling simplifies the image by converting it from a color format (RGB) to gray, reducing the image to a single channel, where each pixel represents an intensity value ranging from black to white.

---

### Reference:
The Complete Guide to Image Preprocessing Techniques in Python | by Maahi Patel | Medium  
[Link to article](https://medium.com/@maahip1304/the-complete-guide-to-image-preprocessing-techniques-in-python-dca30804550c)

---

### Real-World AI Application: Facial Recognition for Automated Attendance

Facial recognition systems are widely used in various applications, from security and surveillance to user authentication and attendance management. In educational and corporate settings, automated attendance systems using facial recognition provide a reliable and efficient alternative to traditional methods, addressing issues of inefficiency and the potential for falsification. By automatically identifying individuals through facial features, these systems can log attendance accurately and in real-time.

#### Problem Addressed:
Manual attendance tracking is prone to inefficiencies such as time consumption, especially in large groups, and is susceptible to falsification, where individuals can mark attendance for others. Facial recognition automates this process, reducing human error and eliminating the possibility of proxy attendance.

---

### Image Processing Implementation:
**Facial Recognition for Automated Attendance**

1. **Face Detection:**  
   OpenCV’s res10_300x300_ssd_iter_140000.caffemodel with deploy.prototxt config file was used to perform face detection on frames extracted from the camera input.

   ```python
   # Face detection using OpenCV
  import cv2
import numpy as np
from PIL import Image, ImageTk

# Configuration for DNN model
MODEL_FILE = "../MODELS/res10_300x300_ssd_iter_140000.caffemodel"  # Path to the model file
CONFIG_FILE = "../MODELS/deploy.prototxt"  # Path to the configuration file

# Configuration for Video Source (0 for the default webcam)
video_source = 0
vid = cv2.VideoCapture(video_source)

# Load the DNN model from Caffe
net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)

# Read the frame from the video source
ret, frame = vid.read()

if ret:
    # Resize the frame to fit the DNN input size
    frame_resized = cv2.resize(frame, (300, 300))
    
    # Prepare the image as input for the model
    blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Set the blob as input to the network
    net.setInput(blob)
    
    # Perform forward pass to get face detections
    detections = net.forward()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Only proceed if confidence is greater than 0.5
        if confidence > 0.5:
            # Extract the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x2, y2) = box.astype("int")
            
            # Extract the face region
            face = frame[y:y2, x:x2]
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    
    # Convert frame to RGB and display using Tkinter's ImageTk
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


