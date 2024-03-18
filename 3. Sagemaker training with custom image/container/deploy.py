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
parser.add_argument('endpoint_type', type=str, help='The endpoint configuration (serverless or realtime)')
parser.add_argument('model_name', type=str, help='The name given to the model')
args = parser.parse_args()

conf = args.endpoint_type
model_name = args.model_name

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

image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/deployenv:latest"
model_output_path = '<enter s3 preffix>'

# Step 1: Submit a training job to sagemaker training job
training_response = client.create_training_job(
    TrainingJobName=model_name,
    HyperParameters={
        'test':'10'
    },
    AlgorithmSpecification={
        'TrainingImage': image_uri,
        'TrainingInputMode': 'File',

    },
    RoleArn=role,
    InputDataConfig=[
        {
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource':{
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://training-data-bucket-sagemaker/',
                    'S3DataDistributionType': 'FullyReplicated'
                }  
            },
            'InputMode': 'File'
        }
    ],
    OutputDataConfig={
        'S3OutputPath': model_output_path
    },
    ResourceConfig={
        'InstanceType': 'ml.m4.xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 1,
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 300
    },
    EnableNetworkIsolation=False,
    EnableManagedSpotTraining=False,
    Environment={
        'someenvval': 'someval'
    },
    RetryStrategy={
        'MaximumRetryAttempts': 1
    }
)

# Check if the model training is complete
while client.describe_training_job(TrainingJobName=model_name)['TrainingJobStatus'] == 'InProgress':
    print('Model training in progress')
    time.sleep(20)

print(client.describe_training_job(TrainingJobName=model_name)['TrainingJobStatus'])

# Step 2: Create a Sagemaker model using the trained model
create_model_response = client.create_model(
    ModelName=model_name,
    Containers=[
        {
            "Image": image_uri,
            "Mode": "SingleModel",
            "ModelDataUrl": f"{model_output_path}{model_name}/output/model.tar.gz"
        }
    ],
    ExecutionRoleArn=role,
)
print("Model Arn: " + create_model_response["ModelArn"])

# Step 3: EPC Creation
if(conf == "serverless"):
    prodVariants = [
        {
            "VariantName":"imageVariant",
            "ModelName": model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": 2048,
                "MaxConcurrency": 20,
                # "ProvisionedConcurrency": 10,
            }
        },
    ]
else:
    prodVariants = [
        {
            "VariantName":"imageVariant",
            "ModelName": model_name,
            "InstanceType": "ml.c5.large",
            "InitialInstanceCount": 1
        },
    ]

endpoint_config_response = client.create_endpoint_config(
    EndpointConfigName=model_name,
    ProductionVariants=prodVariants,
)

print("Endpoint Configuration Arn: " + endpoint_config_response["EndpointConfigArn"])

# Step 3: EP Creation
create_endpoint_response = client.create_endpoint(
    EndpointName=model_name,
    EndpointConfigName=model_name,
)
print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])


#Monitor creation
describe_endpoint_response = client.describe_endpoint(EndpointName=model_name)
while describe_endpoint_response["EndpointStatus"] == "Creating":
    describe_endpoint_response = client.describe_endpoint(EndpointName=model_name)
    print(describe_endpoint_response["EndpointStatus"])
    time.sleep(15)
print(describe_endpoint_response)

