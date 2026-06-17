"""
03 - Question answering over a PDF in S3 (Converse, s3Location source)
Assembled from slide 23 (+ the call pattern from slide 22).

Instead of sending bytes, point Converse at an object in S3. Bedrock must
have read access to the bucket/object.

Edit S3_URI / BUCKET_OWNER below, then:
  python 03_qa_converse_s3_pdf.py
"""

import json
import boto3

REGION = "us-east-1"
MODEL_ID = "us.amazon.nova-lite-v1:0"

S3_URI = "s3://demo-bucket/document1.pdf"   # <-- change me
BUCKET_OWNER = "123456789012"               # <-- change me (account ID)
QUESTION = "Describe the following document"


def main():
    messages = [{
        "role": "user",
        "content": [
            {
                "document": {
                    "format": "pdf",
                    "name": "sample_doc",
                    "source": {
                        "s3Location": {
                            "uri": S3_URI,
                            "bucketOwner": BUCKET_OWNER,
                        }
                    },
                }
            },
            {"text": QUESTION},
        ],
    }]

    inf_params = {"maxTokens": 300, "topP": 0.1, "temperature": 0.3}
    client = boto3.client("bedrock-runtime", region_name=REGION)

    model_response = client.converse(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig=inf_params,
    )

    print("\n[Response Content Text]")
    print(model_response["output"]["message"]["content"][0]["text"])


if __name__ == "__main__":
    main()
