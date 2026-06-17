"""
02 - InvokeModel (non-streaming)
Uses the Amazon Bedrock DATA PLANE ("bedrock-runtime" client).

This is the MODEL-NATIVE format: the request body uses Anthropic's own
schema (anthropic_version, max_tokens, messages). The body shape changes
if you switch to a non-Anthropic model.

Run:  python 02_invoke_model.py
"""

import boto3
import json

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# Native Anthropic request body
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 5000,
    "messages": [
        {
            "role": "user",
            "content": "Create a script to resize images"
        }
    ]
})

response = bedrock_runtime.invoke_model(
    body=body,
    modelId="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    accept="application/json",
    contentType="application/json",
)

# Parse the streamed body object and pull out the text
response_body = json.loads(response.get("body").read())
print("Response from the model:\n")
print(response_body["content"][0]["text"])
