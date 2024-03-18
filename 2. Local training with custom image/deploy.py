import boto3
import json
import os
import joblib
import pickle
import tarfile
import sagemaker
from sagemaker.estimator import Estimator
import argparse
import time
from time import gmtime, strftime
import subprocess

# Read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='The endpoint configuration (serverless or realtime)')
args = parser.parse_args()

conf = args.input

client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")
sts = boto3.client('sts')

account = sts.get_caller_identity().get('Account')

boto_session = boto3.session.Session()
s3 = boto_session.resource('s3')

region = boto_session.region_name
print(region)

sagemaker_session = sagemaker.Session()
role = "<enter arn>"

#  Build the docker image and push to ecr
bashCommand = "./build_and_push.sh deployenv"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#  Retrieve the image we just uploaded to aws
image = "deployenv"
image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{image}:latest"

#Step 1: Model Creation
model_name = "sklearn-test" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Model name: " + model_name)
create_model_response = client.create_model(
    ModelName=model_name,
    Containers=[
        {
            "Image": image_uri,
            "Mode": "SingleModel",
        }
    ],
    ExecutionRoleArn=role,
)
print("Model Arn: " + create_model_response["ModelArn"])

#Step 2: EPC Creation
sklearn_epc_name = "sklearn-epc" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

if(conf == "serverless"):
    endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName=sklearn_epc_name,
        ProductionVariants=[
            {
                "VariantName":"imageVariant",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": 2048,
                    "MaxConcurrency": 20,
                    # "ProvisionedConcurrency": 10,
                }
            },
        ],
    )
else:
    endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName=sklearn_epc_name,
        ProductionVariants=[
            {
                "VariantName":"imageVariant",
                "ModelName": model_name,
                "InstanceType": "ml.c5.large",
                "InitialInstanceCount": 1
            },
        ],
    )
print("Endpoint Configuration Arn: " + endpoint_config_response["EndpointConfigArn"])

# # Step 3: EP Creation
endpoint_name = "sklearn-local-ep" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
create_endpoint_response = client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=sklearn_epc_name,
)
print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])


#Monitor creation
describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)
while describe_endpoint_response["EndpointStatus"] == "Creating":
    describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)
    print(describe_endpoint_response["EndpointStatus"])
    time.sleep(15)
print(describe_endpoint_response)

