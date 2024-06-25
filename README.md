# Business Problem:
AutoVehicle is a leading self-driving car manufacturing company that is committed to ensuring the safety of its autonomous vehicles on the road. To achieve this goal, the company has developed a proprietary dataset that contains images of objects on the road, captured by the sensors on its autonomous vehicles while customers are driving the cars.

AutoVehicle has hired a team of data scientists to build a machine learning model that can accurately detect hazardous and non-hazardous vehicles on the road using this proprietary dataset. The team of data scientists includes Boise, Mellisa, and Jeevan, who are all experts in the field of computer vision and deep learning.

The team began by exploring the proprietary dataset and performing various preprocessing steps to clean and prepare the data for training. Once the dataset was preprocessed, the team used ML models to train a machine learning model that could accurately detect hazardous and non-hazardous vehicles on the road. 

After several hours of training the model, the team achieved high accuracy on the validation set. The model was then deployed in a production environment, where it was able to detect hazardous and non-hazardous vehicles on the road in real-time.

AutoVehicle was thrilled with the results of the machine learning model, as it significantly improved the safety of its autonomous vehicles on the road. The company's customers were also pleased with the increased safety features of the self-driving cars, which helped to boost customer satisfaction and loyalty.

# Importance of reliable hazardous object detection for self-driving cars

Reliable hazardous object detection is a critical component of self-driving cars, as it plays a crucial role in ensuring the safety of autonomous vehicles on the road. Hazardous objects such as other cars, trucks etc can be detected using a variety of sensors such as cameras and passed to the object detection models to identify the objects. In this project, we will use a pre-trained object detection model to detect hazardous objects in images.

# Data Understanding


The dataset is divided into two categories: training and validation. 

The training dataset consists of 18,000 unique images, and the validation dataset consists of 4,241 distinct images. On average, each image in the validation dataset contains seven objects, each belonging to one of five classes: Car, Truck, Pedestrian, Bicycle, and Traffic Light.

For each image in the training dataset, the bounding box coordinates of the object(s) present in the image are provided in the format of (xmin, xmax, ymin, ymax). These coordinates are used to draw the bounding box around the object in the image, which defines the region of interest around the object and is essential for object detection and localization.

The validation dataset provides a separate set of images that can be used to evaluate the performance of the machine learning model. The goal is to develop a model that can accurately detect and classify objects in new, unseen images, which is essential for the safe and reliable operation of autonomous vehicles on the road. The bounding box coordinates provided in the dataset enable the creation of comprehensive object detection algorithms that can detect and avoid potential hazards on the road.

## Eexplanatory Data Analysis

 ```python
    # Define the path to the labels file  
    labels_file = os.path.join(project_root, 'data', 'labels.csv')  
    # Read the labels file into a pandas dataframe  
    df = pd.read_csv(labels_file)
    df.head()
  ```
## Plot the distribution of the bounding box coordinates and class_id
  ```python

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    df['xmin'].hist(ax=axs[0], bins=20)
    axs[0].set_title('xmin')
    df['ymin'].hist(ax=axs[1], bins=20)
    axs[1].set_title('ymin')
    df['xmax'].hist(ax=axs[2], bins=20)
    axs[2].set_title('xmax')
    df['ymax'].hist(ax=axs[3], bins=20)
    axs[3].set_title('ymax')
    df['class_id'].hist(ax=axs[4], bins=20)
    axs[4].set_title('class_id')
    plt.show()
   ```
## Plot the image sizes
   ```python
   # Get the image sizes
   image_sizes = []
   for i in range(len(image_files)):
       img = Image.open(image_files[i])
       image_sizes.append(np.array(img).shape[:2])
   
   # Get the aspect ratios
   aspect_ratios = [size[0] / size[1] for size in image_sizes]
   
   # Plot the image sizes
   fig, axs = plt.subplots(1, 2, figsize=(20, 5))
   axs[0].hist([size[0] for size in image_sizes], bins=20)
   axs[0].set_title('Image Heights')
   axs[1].hist([size[1] for size in image_sizes], bins=20)
   axs[1].set_title('Image Widths')
   plt.show()
   ```
## EDA on the Image data

   ```python
   # Visualize the bounding boxes in the image
   img_file = '1479505925975418095.jpg'
   img = cv2.imread(os.path.join(download_directory, 'images', img_file))
   frame_np  = np.array(frames[img_file], dtype=object)
   draw_img = draw_boxes(img, frame_np[:,0], draw_dot=True)
   plt.figure(figsize=(10, 10))
   plt.imshow(draw_img)
   plt.axis('off')
   plt.show()
  ```
## Data Processing, Preprocessing & Feature Engineering
   ```python
    # function to resize the image
   def resize_image(image, max_size=244):
       # Get the scale factor
       scale_factor = max_size / max(image.shape[0], image.shape[1])
       # Resize the image
       resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
       return resized_image
   
   # function to convert the image to RGB
   def convert_to_rgb(image):
       rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       return rgb_image
   
   # function to normalize the image
   def normalize_image(image):
       normalized_image = image / 244.0
       return normalized_image
   
   # function to add a dimension to the image
   def add_dimension(image):
       expanded_image = np.expand_dims(image, 0)
       return expanded_image
   
   # function to preprocess the images
   def preprocess_images(images):
       preprocessed_images = []
       for image in images:
           resized_image = resize_image(image)
           rgb_image = convert_to_rgb(resized_image)
           normalized_image = normalize_image(rgb_image)
           expanded_image = add_dimension(normalized_image)
           preprocessed_images.append(expanded_image)
       return preprocessed_images
   
   # take first 5 images and preprocess them
   images = [cv2.imread(image_file) for image_file in image_files[:5]]
   
   preprocessed_images = preprocess_images(images)
   
   # display the preprocessed images
   for image in preprocessed_images:
       plt.figure()
       plt.imshow(image[0])
       plt.axis('off')
  ```

## 
