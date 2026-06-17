"""
03 - InvokeModelWithResponseStream (streaming)
Same native body as InvokeModel, but tokens arrive in chunks.

Each event is a JSON object with a "type". We only print the text deltas:
  message_start -> content_block_start -> content_block_delta (text) ...
  -> content_block_stop -> message_delta -> message_stop

Run:  python 03_invoke_model_stream.py
"""

import boto3
import json

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

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

response = bedrock_runtime.invoke_model_with_response_stream(
    body=body,
    modelId="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    accept="application/json",
    contentType="application/json",
)

print("Streaming invoke response:\n")
stream = response.get("body")
for event in stream:
    chunk = event.get("chunk")
    if chunk:
        chunk_obj = json.loads(chunk.get("bytes").decode())
        if chunk_obj["type"] == "content_block_delta":
            print(chunk_obj["delta"]["text"], end="", flush=True)
print()  # trailing newline
