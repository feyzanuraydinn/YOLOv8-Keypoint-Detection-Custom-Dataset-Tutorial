# YOLOv8 Keypoint Detection on a Custom Dataset Tutorial

## What is Keypoint Detection?

Keypoint detection is a fundamental computer vision task that involves identifying and localizing specific points of interest within an image. These points, also referred to as keypoints or landmarks, can represent various object parts, such as facial features, joints in a human body, or points on animals. Keypoint detection plays a crucial role in tasks like human pose estimation, facial expression analysis, hand gesture recognition, and more.

## Tutorial Overview

In this tutorial, we will explore the keypoint detection step by step by harnessing the power of YOLOv8, a state-of-the-art object detection architecture. Our journey will involve crafting a custom dataset and adapting YOLOv8 to not only detect objects but also identify keypoints within those objects. This endeavor opens the door to a wide array of applications, from human pose estimation to animal part localization, highlighting the versatility and impact of combining advanced detection techniques with the precision of keypoint identification. We will undertake the practical aspect of this tutorial by constructing a custom dataset centered around animals, enabling us to implement keypoint detection on these subjects.

## Installation

You'll need a working installation of Python to run the code examples and scripts. The code in this tutorial is written in Python 3.11.4.
Install required libraries using pip:

```
pip install -r requirements.txt
```

## Dataset

To get started, we will make use of the renowned AwA dataset, which comprises a meticulously curated assortment of images showcasing a wide array of animal species. To access the dataset, please proceed to the official [AwA dataset](https://cvml.ista.ac.at/AwA2/) website.

## Data Annotation

Annotate the collected images or videos with bounding boxes and corresponding class labels. There are various annotation tools available, such as [LabelImg](https://pypi.org/project/labelImg/), [RectLabel](https://rectlabel.com), [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/), [CVAT](https://www.cvat.ai) etc.
In this tutorial, we will employ [CVAT](https://www.cvat.ai) for our data annotation needs. To facilitate the annotation process, follow the steps outlined below:

1. Go to [CVAT](https://www.cvat.ai).
2. Click on 'Start using CVAT', and on the redirected page, create an account and log in.
3. Navigate to the 'Projects' section, click the '+' button, and select 'Create a new project'. Enter the project name, for example: 'keypoint-detection', and add the label as 'quadruped' (as we'll be working with quadruped animals). Click 'Continue', then 'Submit & Open'. The opened screen will prompt you to add all necessary labels, but for this project, we'll use only one label, which we added in the previous step.
4. Click the '+' button, select 'Create a new task', name the task, and then add the photos from the downloaded dataset to the task. Click 'Select & Continue'. At this point, you might need to wait a while for the data to upload to the server. Once the upload is complete, go to the 'Tasks' section and select the task you created.
5. At this stage, we'll begin the annotation process. To do this, select the 'Draw New Points' tool and enter the number of keypoints you'll be annotating. In this tutorial, we'll annotate 39 keypoints. After entering '39', click 'Shape' and you're ready to proceed. However, note that it's important to perform keypoint annotation in a specific order. For the exact order of keypoints used in this tutorial, you can refer to 'order.txt'.
   ![keypoints](https://github.com/prinik/AwA-Pose/blob/main/Images/sample.png?raw=true)
   Following this order that you've defined for each photograph, you must proceed with the marking process. For each of the 39 keypoints, ensure you annotate them accordingly. After marking all keypoints, utilize the 'Draw a Rectangle' tool to draw a bounding box around the object. Make sure to apply these steps across the entire dataset.
6. Once you've completed the annotation process, save your changes and navigate to the 'Tasks' section. Click on the 'Actions' button associated with your task, then select 'Export Task Dataset'. Choose 'CVAT for Images 1.1' as the export format, and confirm by selecting 'OK' to initiate the export process. After the download process is complete, place the obtained 'annotation.xml' file in the same directory as your project.

7. Next, you need to convert the data into the format required by YOLOv8. To achieve this, run the 'CVAT_to_cocokeypoints' script, which will facilitate the conversion process.

## Data Format and File System
Before you train YOLOv8 with your dataset you need to be sure if your dataset file format is proper.
In the images directory there are our annotated images (.jpg) that we download before and in the labels directory there are annotation label files (.txt) which has the same names with related images. Just like this:

<details open><summary>data</summary><blockquote>
        <details open><summary>images<summary><blockquote>
        <details open><summary>train<summary><blockquote>
        image_1.jpg <br>
        image_2.jpg <br>
        image_3.jpg <br>
        </blockquote></details>
        <details open><summary>val<summary><blockquote>
        image_x.jpg <br>
        image_y.jpg <br>
        image_z.jpg <br>
        </blockquote></details>
        </blockquote></details>
        <details open><summary>labels<summary><blockquote>
        <details open><summary>train<summary><blockquote>
        image_1.txt <br>
        image_2.txt <br>
        image_3.txt <br>
        </blockquote></details>
        <details open><summary>val<summary><blockquote>
        image_x.txt <br>
        image_y.txt <br>
        image_z.txt <br>
        </blockquote></details>
        </blockquote></details>
</blockquote></details>

You can download the dataset used to train the model in this tutorial from [here](https://drive.google.com/drive/folders/1KynwMSFcAluBFjCnTldN8D0OWSuyvKgW).

## Train Model
We need a configuration (.yaml) file with the same directory as our project. The configuration file (config.yaml) is a crucial component that provides necessary information to customize and control the training process of your keypoint detection model using the YOLOv8 architecture. 
This simple configuration file have a few keys which are: path, train, val, names, kpt_shape and flip_idx. 

**kpt_shape:** This indicates the number of keypoints (39) and the number of dimensions (3) for each keypoint's annotation. The [39, 3] configuration suits your dataset's keypoint format.

**flip_idx:** These are indices that define the order of keypoints for flipping transformations. Flipping might change the left-right orientation of keypoints, so this order ensures consistency.

Make sure to adjust the paths (path, train and val) according to your setup. 

```
#config.yaml
path: path: /home/user/Desktop/yolov8keypoint-detection-tutorial/code/data      # dataset root directory
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Keypoints
kpt_shape: [39, 3]  # [number of keypoints, number of dim]
flip_idx: [0, 1, 2, 4, 3, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 27, 28, 29, 24, 25,
           26, 33, 34, 35, 30, 31, 32, 36, 38, 37]

# Classes
names:
  0: quadruped
```

Once you've configured the config.yaml file, proceed to train the model by running the train.py script.

This concludes the data preparation and configuration steps, enabling us to delve into training our model using the YOLOv8 architecture. We are finally ready to test it.


## Test Model

To test the model, acquire a '.jpg' image file and then execute the 'test.py' script. 
This script uses the ultralytics library to load the YOLOv8 model and perform inference on the image. It then visualizes the keypoints on the image using OpenCV. Make sure to adjust the paths (model_path and image_path) according to your setup. This script provides a basic way to test your trained model and visualize its keypoint predictions on an image.


