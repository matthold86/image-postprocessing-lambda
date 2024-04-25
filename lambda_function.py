import boto3
import cv2
import numpy as np
from io import BytesIO
import logging
import sys
import random
from datetime import datetime

# Initialize Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

# Initialize the S3 clients
s3_client = boto3.client('s3')

def lambda_handler(event, context):

    # Extract object key and object url from stepfunction payload
    predictions  = event['predictions']
    statusCode = event['statusCode']
    raw_objectkey = event['raw_objectkey']
    bucket_name = event['bucket_name']
    image_id = event['image_id']

    # Download raw image from S3
    file_obj = s3_client.get_object(Bucket=bucket_name, Key=raw_objectkey)
    image_file_body = file_obj['Body'].read()
    logger.info(str(datetime.now()) + ": Postprocessing - Downloaded raw image")
    
    # Decode image
    image_array = np.frombuffer(image_file_body, dtype=np.uint8)
    orig_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    logger.info(str(datetime.now()) + ": Postprocessing - Image decoded successfully.")

    # Image Parameters
    image_height, image_width, _ = orig_image.shape
    model_height, model_width = 300, 300
    x_ratio = image_width/model_width
    y_ratio = image_height/model_height
    logger.info(str(datetime.now()) + ": Postprocessing - Extracted image parameters.")

    # Image Postprocessing using inference results
    if 'boxes' in predictions:
        for idx,(x1,y1,x2,y2,conf,lbl) in enumerate(predictions['boxes']):
            # Draw Bounding Boxes
            x1, x2 = int(x_ratio*x1), int(x_ratio*x2)
            y1, y2 = int(y_ratio*y1), int(y_ratio*y2)
            color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
            cv2.rectangle(orig_image, (x1,y1), (x2,y2), color, 4)
            cv2.putText(orig_image, f"Class: {int(lbl)}", (x1,y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(orig_image, f"Conf: {int(conf*100)}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            if 'masks' in predictions:
                # Draw Masks
                mask = cv2.resize(np.asarray(predictions['masks'][idx]), dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
                for c in range(3):
                    orig_image[:,:,c] = np.where(mask>0.5, orig_image[:,:,c]*(0.5)+0.5*color[c], orig_image[:,:,c])
        logger.info(str(datetime.now()) + ": Postprocessing - Processed boxes.")
    if 'probs' in predictions:
        # Find Class
        lbl = predictions['probs'].index(max(predictions['probs']))
        color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
        cv2.putText(orig_image, f"Class: {int(lbl)}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        logger.info(str(datetime.now()) + ": Postprocessing - Processed probabilities.")

    if 'keypoints' in predictions:
        # Define the colors for the keypoints and lines
        keypoint_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
        line_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))

        # Define the keypoints and the lines to draw
        # keypoints = keypoints_array[:, :, :2]  # Ignore the visibility values
        lines = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Torso
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Draw the keypoints and the lines on the image
        for keypoints_instance in predictions['keypoints']:
            # Draw the keypoints
            for keypoint in keypoints_instance:
                if keypoint[2] == 0:  # If the keypoint is not visible, skip it
                    continue
                cv2.circle(orig_image, (int(x_ratio*keypoint[:2][0]),int(y_ratio*keypoint[:2][1])), radius=5, color=keypoint_color, thickness=-1)

            # Draw the lines
            for line in lines:
                start_keypoint = keypoints_instance[line[0]]
                end_keypoint = keypoints_instance[line[1]]
                if start_keypoint[2] == 0 or end_keypoint[2] == 0:  # If any of the keypoints is not visible, skip the line
                    continue
                cv2.line(orig_image, (int(x_ratio*start_keypoint[:2][0]),int(y_ratio*start_keypoint[:2][1])),(int(x_ratio*end_keypoint[:2][0]),int(y_ratio*end_keypoint[:2][1])), color=line_color, thickness=2)
        logger.info(str(datetime.now()) + ": Postprocessing - Processed keypoints.")

    # Store image in S3 database
    image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".jpg", image_rgb)
    io_buf = BytesIO(buffer)

    target_key = "postprocessed-images/" + image_id + ".jpg"
    s3_client.put_object(Bucket=bucket_name, Key=target_key, Body=io_buf)
    logger.info(str(datetime.now()) + f": Postprocessing - Successfully post-processed and uploaded the final image to {target_key}")
    
    return {
        "Payload": {
            "bucket_name": bucket_name,
            "postprocessed_objectkey": target_key
        }
    }
