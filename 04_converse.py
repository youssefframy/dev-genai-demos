"""
04 - Converse (unified API, non-streaming)
The Converse API gives ONE request/response shape across all models.
Swap the modelId and the rest of the code stays the same.

  - messages / system: standardized prompt structure
  - inferenceConfig: standardized params (temperature, topP, maxTokens)
  - additionalModelRequestFields: model-specific params (e.g. top_k)

Run:  python 04_converse.py
"""

import boto3

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock_client.converse(
    modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[
        {
            "role": "user",
            "content": [{"text": "Create a script to resize images."}],
        }
    ],
    system=[
        {"text": "You are an app developer proficient in Python. "
                 "Only engage in discussion of coding topics."}
    ],
    inferenceConfig={"temperature": 0.7, "topP": 0.9, "maxTokens": 500},
    additionalModelRequestFields={"top_k": 200},
)

# Unified, model-agnostic output shape
print("Response from the model:\n")
print(response["output"]["message"]["content"][0]["text"])

usage = response["usage"]
print(f"\n(tokens — in: {usage['inputTokens']}, "
      f"out: {usage['outputTokens']}, total: {usage['totalTokens']})")
