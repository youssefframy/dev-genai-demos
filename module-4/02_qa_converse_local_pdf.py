"""
02 - Question answering over a local PDF (Converse, document bytes)
Assembled from slides 21-22.

Reads a PDF from disk, passes its bytes in the Converse `document` block,
and asks a question about it.

Run:  python 02_qa_converse_local_pdf.py path/to/my_document.pdf "Your question?"
"""

import sys
import json
import boto3

REGION = "us-east-1"
MODEL_ID = "us.amazon.nova-lite-v1:0"

DEFAULT_QUESTION = (
    "How many qubits of growth is projected by 2026 by the industry, "
    "and how does the actual trajectory differ?"
)


def main(pdf_path, question):
    with open(pdf_path, "rb") as file:
        doc_bytes = file.read()

    messages = [{
        "role": "user",
        "content": [
            {
                "document": {
                    "format": "pdf",
                    "name": "DocumentPDFmessages",
                    "source": {"bytes": doc_bytes},
                }
            },
            {"text": question},
        ],
    }]

    inf_params = {"maxTokens": 300, "topP": 0.1, "temperature": 0.3}
    client = boto3.client("bedrock-runtime", region_name=REGION)

    model_response = client.converse(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig=inf_params,
    )

    print("\n[Full Response]")
    print(json.dumps(model_response, indent=2, default=str))
    print("\n[Response Content Text]")
    print(model_response["output"]["message"]["content"][0]["text"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 02_qa_converse_local_pdf.py <pdf_path> [question]")
        raise SystemExit(1)
    path = sys.argv[1]
    q = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUESTION
    main(path, q)
