"""
01 - Text generation: sentiment test data (Generation pattern)
Reconstructed and completed from slides 12-15.

Generates synthetic city reviews with Amazon Nova Lite and (optionally)
stores them in DynamoDB. DynamoDB write is OFF by default so the script
runs standalone — flip USE_DYNAMODB to True once you have a table.

Run:  python 01_text_generation_sentiment.py
"""

import json
import random
import boto3

# ---- Config -----------------------------------------------------------------
REGION = "us-east-1"
MODEL_ID = "us.amazon.nova-lite-v1:0"
CITIES = ["Seattle", "Austin", "Albuquerque", "Portland", "Denver"]

USE_DYNAMODB = False                      # set True to persist results
DYNAMODB_TABLE = "city-reviews"           # only used when USE_DYNAMODB is True

bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)


# ---- Prompt + rating builder (slide 13, all branches filled in) -------------
def generate_review_prompt(city, sentiment):
    if sentiment == "positive":
        prompts = [
            f"Write an enthusiastic 4- or 5-star review of {city}. Focus on what "
            f"makes this city amazing - the food scene, culture, attractions, and "
            f"lifestyle. Be specific about places and experiences. Write 500-700 "
            f"words as someone who loves living here.",
            f"Create a glowing review of {city} highlighting the best aspects - "
            f"great neighborhoods, friendly people, excellent dining, and unique "
            f"attractions. Make it personal and positive. 500-700 words.",
        ]
        rating = random.choice([4, 4, 5])          # more 4s than 5s
    elif sentiment == "negative":
        prompts = [
            f"Write a critical 1- or 2-star review of {city}. Focus on specific "
            f"frustrations - cost of living, traffic, weather, or service. Stay "
            f"realistic and concrete. 500-700 words.",
            f"Create a disappointed review of {city} describing what fell short "
            f"for you as a resident or visitor. Be specific and fair. 500-700 words.",
        ]
        rating = random.choice([1, 2, 2])
    else:  # neutral
        prompts = [
            f"Write a balanced 3-star review of {city} weighing the good against "
            f"the bad. Mention concrete pros and cons. 500-700 words.",
            f"Create an even-handed review of {city} - some things work, some "
            f"don't. Keep it specific and measured. 500-700 words.",
        ]
        rating = 3

    return random.choice(prompts), rating


# ---- Sentiment mix (referenced on slide 12) ---------------------------------
def generate_sentiment_distribution():
    """Roughly 50% positive, 25% neutral, 25% negative, one per city."""
    pool = (["positive"] * 5) + (["neutral"] * 2) + (["negative"] * 3)
    return [random.choice(pool) for _ in CITIES]


# ---- Model call (slides 14-15) ----------------------------------------------
def call_nova(prompt, bedrock_client, model_id=MODEL_ID):
    """Call Amazon Nova Lite to generate a review."""
    try:
        message_list = [{"role": "user", "content": [{"text": prompt}]}]
        inf_params = {"maxTokens": 1000, "temperature": 0.7}
        body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "inferenceConfig": inf_params,
        }
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
        )
        response_body = json.loads(response["body"].read())
        return response_body["output"]["message"]["content"][0]["text"]
    except Exception as exc:  # keep the loop alive on a single failure
        print(f"  ! model call failed: {exc}")
        return None


# ---- Optional persistence (slide 12 referenced upload_to_dynamodb) ----------
def upload_to_dynamodb(review_text, rating, city, table):
    import time
    table.put_item(Item={
        "city": city,
        "timestamp": int(time.time()),
        "rating": rating,
        "review": review_text,
    })
    return True


# ---- Driver (slide 12) ------------------------------------------------------
def main():
    table = None
    if USE_DYNAMODB:
        table = boto3.resource("dynamodb", region_name=REGION).Table(DYNAMODB_TABLE)

    sentiments = generate_sentiment_distribution()

    for i, city in enumerate(CITIES):
        print(f"\nProcessing {i + 1}: {city} ({sentiments[i]})")
        prompt, rating = generate_review_prompt(city, sentiments[i])
        review_text = call_nova(prompt, bedrock_client)
        if review_text:
            print(f"  rating={rating}, {len(review_text.split())} words")
            print(f"  preview: {review_text[:160]}...")
            if USE_DYNAMODB and upload_to_dynamodb(review_text, rating, city, table):
                print("  stored in DynamoDB")


if __name__ == "__main__":
    main()
