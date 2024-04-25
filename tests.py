import boto3
from moto import mock_aws
from lambda_function import lambda_handler  # Ensure your lambda function is in a Python module
import unittest

@mock_aws
class TestLambdaFunction(unittest.TestCase):
    def setUp(self):
        # Set up the moto mock
        self.mock_s3 = mock_aws()
        self.mock_s3.start()

        # Create a mock S3 bucket and object
        self.s3 = boto3.client('s3', region_name='us-east-1')
        self.bucket_name = 'test-bucket'
        self.s3.create_bucket(Bucket=self.bucket_name)
        self.test_key = 'test_folder/test_image.jpg'
        self.test_url = f'https://s3.amazonaws.com/{self.bucket_name}/{self.test_key}'
        
        # Load a test image and upload to mock S3
        with open("C:\\Users\\matth\\OneDrive\\Pictures\\tetons.jpg", "rb") as f:
            self.s3.put_object(Bucket=self.bucket_name, Key=self.test_key, Body=f.read())

    def tearDown(self):
        self.mock_s3.stop()

    def test_lambda_handler(self):
        event = {
            "object_key": self.test_key,
            "object_url": self.test_url,
            "bucket": self.bucket_name
        }
        context = {}
        response = lambda_handler(event, context)
        print(response)
        # Add assertions to check for the expected output

if __name__ == '__main__':
    unittest.main()