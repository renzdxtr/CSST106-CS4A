### Topic 1.2: Image Processing Techniques
# Machine Problem No. 2: Applying Image Processing Techniques

## Hands-On Exploration
* Lab Session 1: Image Transformations
  * Scaling and Rotation: Learn how to apply scaling and rotation transformations to images using OpenCV.
  * Implementation: Practice these transformations on sample images provided in the lab.

    ```python
    # Scaling
    def scale_image (img, scale_factor):
      height, width = img.shape [:2]
      scaled_img = cv2.resize (img, (int(width * scale_factor),
                                        int( height * scale_factor)),
                                        interpolation = cv2.INTER_LINEAR)
      return scaled_img
    
    # Rotate
    def rotate_image ( img , angle ) :
      height, width = img.shape [:2]
      center = (width // 2, height // 2)
      matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
      rotated_img = cv2.warpAffine(img, matrix, (width, height))
      return rotated_img
    
    # Scale image by 0.5
    scaled_image = scale_image(image, 0.5)
    
    # Rotate image by 45 degrees
    rotated_image = rotate_image(image, 90)
    
    display_images([image, scaled_image, rotated_image],
                   ["Original Image", "Scaled Image (50%)",
                    "Rotated Image (90°)"])
    ```

    ![image](https://github.com/user-attachments/assets/1ef96b59-1c43-434a-ba2f-9e937939c2a0)

* Lab Session 2: Filtering Techniques
  * Blurring and Edge Detection: Explore how to apply blurring filters and edge detection algorithms to images using OpenCV.
  * Implementation: Apply these filters to sample images to understand their effects.
 
    ```python
    # Applies a Gaussian blur to the image
    def gaussian_blur(img, ksize=(15, 15)):
        return cv2.GaussianBlur(img, ksize, 0)
    
    # Applies a median blur to the image
    def median_blur(img, ksize=15):
        return cv2.medianBlur(img, ksize)
    
    # Applies a box blur to the image
    def box_blur(img, ksize=(15, 15)):
        return cv2.blur(img, ksize=ksize)
    
    # Applies a bilateral blur to the image
    def bilateral_blur(img, d=9, sigmaColor=75, sigmaSpace=75):
        return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    
    # Applies a motion blur to the image
    def motion_blur(img, size=15, angle=0):
      # Create a kernel for motion blur with the specified size
      kernel_motion_blur = np.zeros((size, size))
    
      # Set the middle row of the kernel to 1, which will be used to blur the image
      kernel_motion_blur[int((size-1)//2), :] = np.ones(size)
    
      # Normalize the kernel by dividing by its size to ensure the output image has the same intensity
      kernel_motion_blur = kernel_motion_blur / size
    
      # Apply the motion blur kernel to the input image using OpenCV's filter2D function
      blurred_img = cv2.filter2D(img, -1, kernel_motion_blur)
    
      # Return the blurred image
      return blurred_img

    gaussian_blur_image = gaussian_blur(image)
    median_blur_image = median_blur(image)
    box_blur_image = box_blur(image)
    bilateral_blur_image = bilateral_blur(image)
    motion_blur_image = motion_blur(image)

    display_images([gaussian_blur_image, 
                median_blur_image, 
                box_blur_image, 
                bilateral_blur_image, 
               motion_blur_image],
               ['Gaussian Blur', 
                'Median Blur', 
                'Box Blur', 
                'Bilateral Blur', 
                'Motion Blur'])
    ```
    ![image](https://github.com/user-attachments/assets/60657d49-9072-47ca-8de4-dc1c899b8f01)

    ```python
    # Applies the Canny edge detection algorithm to the image
    def canny(img, threshold1=100, threshold2=200):
        return cv2.Canny(img, threshold1, threshold2)
    
    # Applies the Sobel edge detection algorithm to the image
    def sobel_edge_detection(img):
        # Sobel edge detection in the x direction
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        # Sobel edge detection in the y direction
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # Combine the two gradients
        grad = np.sqrt(grad_x**2 + grad_y**2)
        # Normalize to [0, 255]
        image = np.uint8(grad / grad.max() * 255)
        return image
    
    # Applies the Laplacian edge detection algorithm to the image
    def laplacian_edge_detection(img):
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        # Normalize to [0, 255]
        image = np.uint8(laplacian / laplacian.max() * 255)
        return image
    
    # Applies the Prewitt edge detection algorithm to the image
    def prewitt_edge_detection(img):
        # Prewitt operator kernels for x and y directions
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        # Applying the Prewitt operator
        grad_x = cv2.filter2D(img, -1, kernel_x)
        grad_y = cv2.filter2D(img, -1, kernel_y)
        # Combine the x and y gradients by converting to floating point
        grad = np.sqrt(grad_x**2 + grad_y**2)
        # Normalize to [0, 255]
        image = np.uint8(grad / grad.max() * 255)
        return image

    canny_image = canny(image, 100, 200)
    sobel_image = sobel_edge_detection(image)
    laplacian_image = laplacian_edge_detection(image)
    prewitt_image = prewitt_edge_detection(image)

    display_images([canny_image, 
                sobel_image, 
                laplacian_image, 
                prewitt_image],
               ['Canny', 
                'Sobel', 
                'Laplacian', 
                'Prewitt'])
    ```
    ![image](https://github.com/user-attachments/assets/13dac6d8-1158-44a1-a2cd-dfe262c5f4a8)

## Problem-Solving Session
* Common Image Processing Tasks
  * Engage in a problem-solving session focused on common challenges encountered in image processing tasks.
  * Scenario-Based Problems: Solve scenarios where you must choose and apply appropriate image processing techniques.

  ### Scenario 1: Scaling and Rotation
   * Problem:  Standardizing product images for an e-commerce website.
   * Explanation: This scenario addresses the common challenge of dealing with product images of varying sizes and orientations in e-commerce. The goal is to create a consistent visual presentation across the website.

     ```python
     # Load the image
     image = cv2.imread('product.jpg')
     
     # Convert BGR image to RGB
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
     # Scale the image
     scaled_image = scale_image(image, 0.5)
     
     # Rotate the scaled image
     rotated_image = rotate_image(scaled_image, 180)
     
     # Save the processed image
     cv2.imwrite('processed_product.jpg', rotated_image)
     
     # Display results
     display_images([image, rotated_image], ["Original Image", "Processed Image"])
     cv2.waitKey(0)
     cv2.destroyAllWindows()
     ```
     ![image](https://github.com/user-attachments/assets/3307db5d-e86b-402d-b739-1afb8d97dc87)
 
     The processed image is saved as well:
      
     ![image](https://github.com/user-attachments/assets/d104dedb-9189-4b22-8a64-e5e2f5094692)

  * Scaling: We use the scale_image function, which employs OpenCV's resize method with bilinear interpolation. This approach provides a good balance between speed and quality for most images. The function allows for both upscaling and downscaling, with a scale factor to determine the new size relative to the original.
  * Rotation: The rotate_image function uses an affine transformation to rotate the image around its center. This is crucial for correcting image orientation. The function can handle any rotation angle, making it versatile for various image orientations.

By combining these operations, we can ensure that all product images are displayed at a consistent size and orientation, improving the overall aesthetic and user experience of the e-commerce platform.

  ### Scenario 2: Blurring (Image Smoothing)
   * Problem: Implementing a skin-softening feature in a photo editing app.
   * Explanation: This scenario tackles the challenge of selectively smoothing skin textures in portrait photos, a common feature in photo editing applications.

     ```python
     # Define a basic skin detection function
     def detect_skin(img):
         # Convert the image to YCrCb color space
         ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
         
         # Define the range of skin color in YCrCb
         lower = np.array([0, 133, 77], dtype=np.uint8)
         upper = np.array([255, 173, 127], dtype=np.uint8)
         
         # Create a binary mask for skin pixels
         skin_mask = cv2.inRange(ycrcb, lower, upper)
         
         # Apply morphological operations to clean up the mask
         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
         skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
         skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
         
         # Blur the mask to smooth out the edges
         skin_mask = gaussian_blur(skin_mask, (3, 3))
         
         return skin_mask
     
     # Define smooth skin function
     def smooth_skin(image, skin_mask):
         blurred = gaussian_blur(image, (3, 3))
         bilateral = bilateral_blur(blurred, d=3, sigmaColor=20, sigmaSpace=20)
     
         result = np.where(skin_mask[..., np.newaxis] == 255, image, bilateral)
         return result
     
     # Usage
     img = cv2.imread('portrait.jpg')
     
     # Convert BGR image to RGB
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     
     if img is None:
         print("Error: Could not read the image file.")
     else:
         skin_mask = detect_skin(img)
         smoothed_img = smooth_skin(img, skin_mask)
     
         # Display results
         display_images([img, skin_mask, smoothed_img], ["Original Image", "Skin Mask", "Smoothed Image"])
         cv2.waitKey(0)
         cv2.destroyAllWindows()
     
         # Save the smoothed image
         cv2.imwrite('smoothed_portrait.jpg', smoothed_img)
         
     ```
     ![image](https://github.com/user-attachments/assets/2787a50a-4f55-43ac-a57f-8493d8395a9a)

     The processed image is saved as well:
    
     ![image](https://github.com/user-attachments/assets/94ec30b9-b848-4fce-8179-b79e9c94d2e9)
    
     * Gaussian Blur: The gaussian_blur function applies a Gaussian filter to the entire image. This creates a general softening effect but can be too strong if applied uniformly.
     * Bilateral Filtering: The bilateral_blur function is an edge-preserving smoothing filter. It's particularly effective for skin smoothing, reducing noise while preserving important edge details.
     * Skin Smoothing: The smooth_skin function combines these blurring techniques with a skin mask. It applies the smoothing only to detected skin regions, preserving facial features and other details.

This approach allows for a natural-looking skin softening effect, enhancing portrait photos without losing important facial characteristics. The combination of Gaussian and bilateral filtering provides a balance between smoothing and detail preservation.

  ### Scenario 3: Document Edge Detection and Cropping
   * Problem: Implementing a document edge detection and cropping feature in a document scanner app.
   * Explanation: This scenario tackles the challenge of selectively detecting and cropping the edges of a document from a scanned image or a photo, a common feature in document scanning applications.
  
       ```python
       def detect_edges(image):
           # Convert the image to grayscale
           gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
           # Apply Gaussian blur to reduce noise
           blurred = gaussian_blur(gray, (5, 5))
       
           # Perform Canny edge detection
           edges = canny(blurred, 50, 150)
       
           return edges
       
       def find_contours(image, edges):
           # Find contours in the edge-detected image
           contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
           # Filter contours to find the largest one (assumed to be the document)
           largest_contour = max(contours, key=cv2.contourArea)
       
           # Approximate the contour to a polygon
           epsilon = 0.02 * cv2.arcLength(largest_contour, True)
           approx = cv2.approxPolyDP(largest_contour, epsilon, True)
       
           return approx
       
       def crop_document(image, contour):
           # Get the bounding box of the contour
           x, y, w, h = cv2.boundingRect(contour)
       
           # Crop the image using the bounding box
           cropped_image = image[y:y+h, x:x+w]
       
           return cropped_image
       
       # Usage
       img = cv2.imread('receipt.jfif')
       
       # Convert BGR image to RGB
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
       if img is None:
           print("Error: Could not read the image file.")
       else:
           # Detect edges
           edges = detect_edges(img)
       
           # Find contours
           contour = find_contours(img, edges)
       
           # Crop the document
           cropped_img = crop_document(img, contour)
       
           # Display results
           display_images([img, edges, cropped_img], ["Original Image", "Edge Detection", "Cropped Document"])
           cv2.waitKey(0)
           cv2.destroyAllWindows()
       
           # Save the cropped document image
           cv2.imwrite('cropped_document.jpg', cropped_img)
       ```
      ![image](https://github.com/user-attachments/assets/4b0c227f-f1aa-475d-b72a-e533de65762b)
   
      The cropped document is saved as well:
   
      ![image](https://github.com/user-attachments/assets/f2cf67da-26dc-43f0-b0aa-83fbe0388bb4)

     * Edge Detection: The detect_edges function converts the image to grayscale, applies Gaussian blur to reduce noise, and then uses the Canny edge detection algorithm to identify edges in the image.
     * Contour Detection: The find_contours function finds contours in the edge-detected image using cv2.findContours. It then filters the contours to find the largest one, which is assumed to be the document. The contour is approximated to a polygon using cv2.approxPolyDP.
     * Document Cropping: The crop_document function calculates the bounding box of the detected contour using cv2.boundingRect, and crops the image using this bounding box to isolate the document content. This approach is simpler and more straightforward compared to using perspective transforms.
