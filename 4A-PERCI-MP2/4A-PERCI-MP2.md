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
