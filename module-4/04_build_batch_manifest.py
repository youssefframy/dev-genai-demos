"""
04 - Build a batch inference JSONL manifest (Summarization, batch mode)
Based on the manifest format on slide 19.

Each line is one record: {"recordId": ..., "modelInput": {...}}. Upload the
resulting file to your S3 input bucket, then submit a model invocation job
(CreateModelInvocationJob) pointing at it.

Run:  python 04_build_batch_manifest.py
Output: batch_input.jsonl
"""

import json

OUTPUT_FILE = "batch_input.jsonl"

SYSTEM_PROMPT = (
    "You are a travel expert AI assistant. Create comprehensive, engaging "
    "city summaries based on user reviews."
)

# In a real job you'd inject the actual reviews where REVIEWS goes.
CITY_REVIEWS = {
    "albuquerque": "{{ REVIEWS for Albuquerque }}",
    "seattle": "{{ REVIEWS for Seattle }}",
    "austin": "{{ REVIEWS for Austin }}",
}


def build_record(record_id, reviews_text):
    user_text = (
        f"Use the following reviews for {record_id.title()} {reviews_text}\n\n"
        "Please provide a well-structured summary that includes:\n"
        "1. Overall impression and sentiment\n"
        "2. Top attractions and recommendations"
    )
    return {
        "recordId": record_id,
        "modelInput": {
            "schemaVersion": "messages-v1",
            "messages": [
                {"role": "user", "content": [{"text": user_text}]}
            ],
            "system": [{"text": SYSTEM_PROMPT}],
            "inferenceConfig": {
                "maxTokens": 500,
                "topP": 0.9,
                "topK": 20,
                "temperature": 0.7,
            },
        },
    }


def main():
    with open(OUTPUT_FILE, "w") as f:
        for city, reviews in CITY_REVIEWS.items():
            record = build_record(city, reviews)
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(CITY_REVIEWS)} records to {OUTPUT_FILE}")
    print("Next: upload to your S3 input bucket and submit a "
          "CreateModelInvocationJob.")


if __name__ == "__main__":
    main()
