"""
01 - List foundation models
Uses the Amazon Bedrock CONTROL PLANE ("bedrock" client).

Run:  python 01_list_foundation_models.py
"""

import boto3

# Control-plane client (management ops: list models, guardrails, jobs, tags)
bedrock = boto3.client("bedrock", region_name="us-east-1")

response = bedrock.list_foundation_models()

# Print a compact view of each model
for model in response["modelSummaries"]:
    print(f"{model['modelId']:<55} "
          f"{model['providerName']:<12} "
          f"in={','.join(model['inputModalities'])} "
          f"out={','.join(model['outputModalities'])}")

print(f"\nTotal models: {len(response['modelSummaries'])}")
