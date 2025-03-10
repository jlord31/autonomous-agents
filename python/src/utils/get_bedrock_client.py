import os
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_bedrock_client():
    """Get or create AWS Bedrock client"""
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=os.environ.get('AWS_REGION', 'eu-west-2')
    )
