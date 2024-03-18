import boto3
import json
import argparse

# Read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input', type=float, help='The input value to test the deployed model endpoint')
args = parser.parse_args()

inference_value = args.input

runtime_client = boto3.client('sagemaker-runtime')
content_type = "application/json"
request_body = {"Inputs": [[inference_value]]}
data = json.loads(json.dumps(request_body))
payload = json.dumps(data)
endpoint_name = "test-endpoint"

response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=payload)
result = json.loads(response['Body'].read().decode())['Outputs']
print(result)