# Module 2: Programming with Amazon Bedrock — Demo Scripts

Ready-to-run Python scripts matching the slide examples.

## Setup

```bash
pip install boto3
```

You need AWS credentials with Bedrock access, configured one of these ways:

```bash
aws configure                 # or export the env vars below
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

Make sure model access for Claude Sonnet 4.5 is enabled in the Bedrock
console (Model access page) in your region.

## Scripts

| File | API | Plane | Streams? |
|------|-----|-------|----------|
| `01_list_foundation_models.py` | `ListFoundationModels` | control (`bedrock`) | — |
| `02_invoke_model.py` | `InvokeModel` | data (`bedrock-runtime`) | no |
| `03_invoke_model_stream.py` | `InvokeModelWithResponseStream` | data | yes |
| `04_converse.py` | `Converse` | data | no |
| `05_converse_stream.py` | `ConverseStream` | data | yes |

```bash
python 01_list_foundation_models.py
python 02_invoke_model.py
python 03_invoke_model_stream.py
python 04_converse.py
python 05_converse_stream.py
```

## Two key ideas for students

**Native vs. unified.** `InvokeModel` uses each model's own body format
(here the Anthropic `messages` schema). `Converse` gives one request/response
shape across every model — change `modelId` and nothing else.

**modelId prefixes** (`global.`, `us.`) are cross-region inference profiles.
Pick the one available in your account/region; drop the prefix to call a
single-region model directly.
