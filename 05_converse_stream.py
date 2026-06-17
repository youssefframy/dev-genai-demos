"""
05 - ConverseStream (unified API, streaming)
Unified shape + token streaming. Best pick when you want low
time-to-first-token AND the freedom to swap models without code changes.

Event types you'll see in the stream:
  messageStart -> contentBlockDelta (text) ... -> contentBlockStop
  -> messageStop -> metadata (usage / latency)

Run:  python 05_converse_stream.py
"""

import sys
import boto3

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock_client.converse_stream(
    modelId="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
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

# Print only the content chunks as they arrive
for event in response["stream"]:
    if "contentBlockDelta" in event:
        chunk = event["contentBlockDelta"]
        sys.stdout.write(chunk["delta"]["text"])
        sys.stdout.flush()
print()  # trailing newline
