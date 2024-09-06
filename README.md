### Topic 1.1: Introduction to Computer Vision and Image Processing
# Machine Problem No. 1: Exploring the Role of Computer Vision and Image Processing in AI

https://github.com/user-attachments/assets/61076eac-8d1e-400b-8e50-b211ee2156ef

# Introduction to Computer Vision and Image Processing
Computer Vision is a field of artificial intelligence (AI) that enables systems to interpret and process visual data from the world. It involves techniques to acquire, process, analyze, and understand images to produce numerical or significant information. Computer vision programs analyze raw images and turn them into useful data by breaking them down into simpler elements [^1][^2].

Most computer vision tasks start with 2D images. While images might seem complex, they are actually just collections of pixels. Each pixel can be represented by a single number (grayscale) or a set of three numbers for color (like 255, 0, 0 for RGB) [^1].

![image](https://github.com/user-attachments/assets/49f297f0-f05b-4468-8733-5cca5f29d9c5)

*The Built In favicon (left) shown in grayscale and (right) the same image with the pixel values overlaid. | Image: Jye Sawtell-Rickson (2022) [^1]*

Once an image is converted into numerical data, algorithms process it [^1].

According to an article by Intel titled "What Is Computer Vision?", Computer vision employs AI to perceive and analyze visual data to optimize processes, enabling proactive and faster situational response times, and increasing business and customer value [^3].

Moreover, image processing is crucial in AI as it enhances, manipulates, and analyzes images to extract meaningful information [^4]. This process is essential for improving the quality of images and making them suitable for further analysis by AI systems.

# Types of Image Processing Techniques
1. **Edge Detection:**  
   Edge detection identifies the boundaries within images, allowing AI systems to recognize objects and their shapes, which is vital for tasks such as object detection and image recognition (Kundu, 2024) [^5].

2. **Resizing:**  
   Resizing involves altering the dimensions of an image (e.g., either by enlarging or shrinking it) to ensure consistency and efficiency. It helps in normalizing the data and reducing computational load [^6].

3. **Grayscaling:**  
   Grayscaling simplifies the image by converting it from a color format (RGB) to gray, reducing the image to a single channel, where each pixel represents an intensity value ranging from black to white [^6].

# Real-World AI Application: Facial Recognition for Automated Attendance
Facial recognition systems are widely used in various applications, from security and surveillance to user authentication and attendance management. In educational and corporate settings, automated attendance systems using facial recognition provide a reliable and efficient alternative to traditional methods, addressing issues of inefficiency and the potential for falsification. By automatically identifying individuals through facial features, these systems can log attendance accurately and in real-time.

### Problem Addressed:
Manual attendance tracking is prone to inefficiencies such as time consumption, especially in large groups, and is susceptible to falsification, where individuals can mark attendance for others. Facial recognition automates this process, reducing human error and eliminating the possibility of proxy attendance.

# Image Processing Implementation: Facial Recognition for Automated Attendance
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
   
2. **Pre-processing for Dataset Creation:**  
   Resizing (images were standardized to 200x200 pixels) → Grayscaling → Data Storage (the processed face data is stored in .pkl files)

   ```python
   # Resizing
   face = cv2.resize(face, (200, 200))  # Resize to ensure consistency

   # Grayscaling
   face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

   # Data Storage
   def save_data(person_name):
     """Serializes and saves collected face data and names to files."""

     DATA_PATH = "data/"  # Path for saving serialized data
   
     faces_data = np.asarray(face_data)
     faces_data = faces_data.reshape(len(face_data), -1)
   
     if not os.path.exists(DATA_PATH):
         os.makedirs(DATA_PATH)
   
     names_file = os.path.join(DATA_PATH, 'names.pkl')
     faces_file = os.path.join(DATA_PATH, 'faces_data.pkl')
   
     # Append or create new data for names
     if os.path.isfile(names_file):
         with open(names_file, 'rb') as f:
             names = pickle.load(f)
         names += [person_name] * len(face_data)
     else:
         names = [person_name] * len(face_data)
   
     with open(names_file, 'wb') as f:
         pickle.dump(names, f)
   
     # Append or create new data for faces
     if os.path.isfile(faces_file):
         with open(faces_file, 'rb') as f:
             faces = pickle.load(f)
         faces = np.append(faces, faces_data, axis=0)
     else:
         faces = faces_data
   
     with open(faces_file, 'wb') as f:
         pickle.dump(faces, f)

4. **Pre-processing for Prediction**  
   The face image (RGB) is resized to 50x50 pixels and flattened to match the features expected by the model used which were 7500 features (50 * 50 * 3 = 7500).

   ```python
   # Face is resized to 50x50 pixels and flattened
   face_resized = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)
   
3. **Model Training**  
   The model used for classification was the KNeighborsClassifier with ‘n_neighbors’ parameter set to ‘1’. It was fitted on ‘FACES’ and ‘LABELS’ stored in .pkl files.

   ```python
   from sklearn.neighbors import KNeighborsClassifier
   import numpy as np
   import pickle

   # Paths to data
   LABELS_PATH = "DATA/names.pkl"
   FACES_PATH = "DATA/faces_data.pkl"

   # Load the model and label data from serialized files
   with open(LABELS_PATH, 'rb') as w:
       LABELS = pickle.load(w)
   with open(FACES_PATH, 'rb') as f:
       FACES = pickle.load(f)

   # Create a KNN classifier and train it with the face data and corresponding labels
   knn = KNeighborsClassifier(n_neighbors=1)
   knn.fit(FACES, LABELS)

   # Save the trained model (optional)
   with open('knn_model.pkl', 'wb') as f:
       pickle.dump(knn, f)

# Model Demonstration
[![Model Demonstration](https://github.com/user-attachments/assets/1296451c-1c0f-4806-98a2-8345638d8f9d)](https://drive.google.com/file/d/1l5K-B0lCw0L-LZEKT9vqizD8X_8DCCL-/preview)

# Conclusion
Effective image processing is important in AI because it helps systems analyze and understand visual data better. Techniques like edge detection, grayscaling, and resizing are crucial for enhancing AI’s ability to recognize, classify, and understand images [^4][^5]. Edge detection identifies object boundaries, helping AI systems detect and distinguish objects, which is vital for tasks like facial recognition [^5]. Grayscaling simplifies images by converting them to a single color channel, making them simpler to process while keeping important details [^6]. Resizing ensures that images conform to the required input dimensions of the model, allowing for efficient and accurate processing [^6].

Looking back at the face recognition model I made for the automated attendance tracker, these techniques were crucial. Edge detection helped the model highlight facial boundaries, aiding the model in recognizing key features, Grayscaling, on the other hand, made images easier to work with, and resizing the images to 50x50 pixels ensured the data fit the model’s input needs. These steps improved the system’s ability to accurately detect and recognize faces, making attendance tracking more reliable.

---
### Extension Activity
# Research on Emerging Form of Image Processing
Recent advancements in image processing have seen a significant shift from traditional machine learning techniques, such as SVMs and k-nearest neighbors, to more sophisticated deep learning (DL) models. These models leverage large neural networks to improve accuracy and efficiency in tasks like face recognition, object detection, and image classification.

Remarkable improvements have been achieved with the introduction of Convolutional Neural Networks (CNNs), currently the backbone of most image analysis systems, which automatically detect important features without any human supervision. Other emerging techniques were the Recurrent Neural Networks (RNNs) and Long Short-Term Memory Networks (LSTMs), which are less common for static images, but significantly useful for sequences like videos. These models can handle applications like video-based face recognition or motion analysis.

The integration of DL into image processing further advances artificial intelligence. These advanced systems can understand and analyze complex visual data with incredible detail. As these technologies continue to develop, they will revolutionize industries by supporting smarter, more adaptable systems that can effectively operate and respond within complex environments.

---
# References:
[^1]: Sawtell-Rickson, J. (2022, December 21). What is Computer Vision? Built In. https://builtin.com/machine-learning/computer-vision
[^2]: DeepAI. (2020, June 25). Computer Vision. DeepAI. https://deepai.org/machine-learning-glossary-and-terms/computer-vision
[^3]: What is Computer Vision? (n.d.). Intel. https://www.intel.com/content/www/us/en/learn/what-is-computer-vision.html
[^4]: GeeksforGeeks. (2024, July 17). AI in Image Processing. GeeksforGeeks. https://www.geeksforgeeks.org/ai-in-image-processing/
[^5]: Kundu, R. (2024, July 25). Image Processing: Techniques, Types, & Applications [2024]. V7. https://www.v7labs.com/blog/image-processing-guide
[^6]: Patel, M. (2023, October 23). The complete guide to image preprocessing techniques in Python. Medium. https://medium.com/@maahip1304/the-complete-guide-to-image-preprocessing-techniques-in-python-dca30804550c
