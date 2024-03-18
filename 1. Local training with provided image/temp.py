import boto3
import sagemaker

client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")

boto_session = boto3.session.Session()
region = boto_session.region_name

# retrieve sklearn image
image_uri = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="0.23-1",
    py_version="py3",
    instance_type="ml.m5.xlarge",
)
print(image_uri)
