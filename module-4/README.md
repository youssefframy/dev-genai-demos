# Module 4: Using Amazon Bedrock APIs in Common Architectures — Demo Scripts

Ready-to-run code assembled from the slide examples. The slides show
fragments (lots of `...`); these are the completed, runnable versions.

## Setup

```bash
pip install boto3
export AWS_DEFAULT_REGION=us-east-1
aws configure        # or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
```

Enable model access for **Amazon Nova Lite** in the Bedrock console
(Model access page) for your region.

## Scripts

| File | Pattern | Notes |
|------|---------|-------|
| `01_text_generation_sentiment.py` | Generation | Nova Lite + optional DynamoDB. Runs standalone. |
| `02_qa_converse_local_pdf.py` | Q&A | Converse with PDF bytes from disk. |
| `03_qa_converse_s3_pdf.py` | Q&A | Converse referencing a PDF in S3. |
| `04_build_batch_manifest.py` | Summarization (batch) | Writes a `batch_input.jsonl` manifest. |
| `05_conversation_lambda.py` | Conversation / memory | Full Lambda handler; needs DynamoDB + API Gateway. |

```bash
python 01_text_generation_sentiment.py
python 02_qa_converse_local_pdf.py my_document.pdf "What is this about?"
python 04_build_batch_manifest.py
```

Scripts 1, 2, and 4 run on a laptop with credentials. Script 3 needs a
real S3 object Bedrock can read. Script 5 is deployed as a Lambda, not run
locally.

## Corrections worth showing students

The conversation-history slides (28 and 32) append a **list** to the
messages array:

```python
conversation_messages.append([{ "role": "user", ... }])   # slide 28
messages.append([{ "role": ..., "content": ... }])          # slide 32
```

That produces a list-of-lists, which Converse rejects — each entry must be
a message **dict**. Script 5 appends dicts instead:

```python
conversation_messages.append({ "role": "user", "content": [...] })
```

Good thing to pause on in class: it's the kind of bug that only surfaces at
runtime against the API.

## Two ideas behind the inference-mode question (knowledge check)

- **Batch** trades latency for cost — async jobs over S3 (the 50-cities-every-
  12-hours scenario). On-demand is the low-latency, pay-per-token path.
- **Memory lives in your frontend/backend**, not in the model or the Bedrock
  APIs. DynamoDB/ElastiCache hold the history; you replay it into each call.
