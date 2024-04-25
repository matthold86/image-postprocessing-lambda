import boto3
import urllib.parse
import cv2
import numpy as np
from io import BytesIO
import logging
import sys
import json

# Initialize the S3 clients
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    # Initialize Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)

    # Extract object key and object url from stepfunction payload
    object_key = event['object_key']
    object_url = event['object_url']
    bucket_name = event['bucket']
    
    logger.info(object_key)
    logger.info(object_url)
    
    # Extract "image_id" from object key
    key_parts = object_key.split('/')
    image_id_with_extension = key_parts[-1]
    image_id = '.'.join(image_id_with_extension.split('.')[:-1])  # Remove the file extension
    logger.info(image_id)

    # Preprocess image
    
    # Download image from S3
    file_obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    image_file_body = file_obj['Body'].read()
    
    # Decode image
    image_array = np.frombuffer(image_file_body, dtype=np.uint8)
    orig_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    logger.info('Image decoded successfully.')
    
    # Preprocess the image
    image_height, image_width, _ = orig_image.shape
    model_height, model_width = 300, 300

    resized_image = cv2.resize(orig_image, (model_width, model_height))
    resized_height, resized_width, _ = resized_image.shape
    _, buffer = cv2.imencode('.jpg', resized_image)
    processed_image = buffer.tobytes()
    logger.info(f'Resized Image: {resized_width}x{resized_height}')
    logger.info(f'Original Image: {image_width}x{image_height}')

    # Upload the processed image to S3
    target_key = "preprocessed-images/" + image_id + ".jpg"
    s3_client.put_object(Bucket=bucket_name, Key=target_key, Body=processed_image)
    
    logger.info(f'Successfully processed and uploaded the image to {target_key}')

    # Structure the response to include this in a 'Payload' key
    response = {
        "Payload": {
            "preprocessed_objectkey": target_key,
            "bucket_name": bucket_name
        }
    }

    # Return the structured response
    return json.dumps(response)