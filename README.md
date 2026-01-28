# CNN -Convolutional Neural Network - Image Feature Extraction and Clustering

This project uses a Convolutional Neural Network (CNN) inspired by VGG to extract feature embeddings from images and group them using KMeans clustering.

## Project Overview

1. **Image Loading and Preprocessing**
   - Read images from a folder and resize them to `(224, 224)`.
   - Convert images to arrays suitable for CNN input.

2. **CNN Feature Extraction**
   - A CNN model similar to VGG extracts visual features from each image.
   - The output is a fixed-length embedding vector representing the image.
<img width="364" height="330" alt="image" src="https://github.com/user-attachments/assets/f2bc73ff-3ecd-4a96-961a-de03f61840b1" />

3. **Clustering**
   - Use KMeans to cluster images based on their embeddings.
   - Similar images are grouped together, enabling dataset organization and visualization.
<img width="605" height="351" alt="image" src="https://github.com/user-attachments/assets/80eba1c1-63d1-424d-a5a8-07d41ac666e8" />

## Usage

```python
# Load images and preprocess
original = load_img(filename, target_size=(224, 224))
numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)
processed_image = imagenet_utils.preprocess_input(image_batch)

# Extract features
predictions = vgg_face_descriptor.predict(processed_image)
list_image.append(predictions[0])

# Cluster using KMeans
estimator = KMeans(n_clusters=3)
estimator.fit(list_image)
label_pred = estimator.labels_
