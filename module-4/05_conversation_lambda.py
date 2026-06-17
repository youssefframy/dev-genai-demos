"""
05 - Conversational memory Lambda (Conversation pattern, long-term memory)
Assembled from the 5-part example on slides 28-32.

Flow per invocation:
  1. Load this user's prior messages from DynamoDB  (get_conversation_history)
  2. Append the new user message and persist it      (store_message)
  3. Call Converse with the full history             (lambda_handler)
  4. Persist the assistant reply                      (store_message)
  5. Return the reply

Deploy as an AWS Lambda. Requires:
  - Env var CONVERSATION_HISTORY_TABLE  -> DynamoDB table name
  - Table key schema: partition key userID (S), sort key timestamp (N)
  - TTL enabled on the 'ttl' attribute
  - Lambda execution role with Bedrock invoke + DynamoDB read/write
  - An HTTP API (API Gateway) with a JWT authorizer supplying the 'sub' claim
"""

import os
import time
import boto3

REGION = "us-east-1"

bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)
dynamodb = boto3.resource("dynamodb")
ddb_client = boto3.client("dynamodb")

TABLE_NAME = os.environ.get("CONVERSATION_HISTORY_TABLE")
conversation_table = dynamodb.Table(TABLE_NAME) if TABLE_NAME else None


# ---- Slides 28-30: handler --------------------------------------------------
def lambda_handler(event, context):
    user_id = event["requestContext"]["authorizer"]["jwt"]["claims"]["sub"]
    conversation_messages = get_conversation_history(user_id)

    user_msg = event["message"]
    conversation_messages.append({
        "role": "user",
        "content": [{"text": user_msg}],
    })
    store_message(user_id, user_msg, "user")

    model_response = bedrock_runtime.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=conversation_messages,
        system=[{
            "text": "Please provide a helpful, conversational response based on "
                    "the available information and conversation history."
        }],
        inferenceConfig={"maxTokens": 300, "temperature": 0.7, "topP": 0.9},
    )

    assistant_msg = model_response["output"]["message"]["content"][0]["text"]
    store_message(user_id, assistant_msg, "assistant")

    return {"statusCode": 200, "body": assistant_msg}


# ---- Slide 31: write one message --------------------------------------------
def store_message(user_id, message, role):
    now_in_seconds = int(time.time())
    expire_ttl = now_in_seconds + (30 * 24 * 60 * 60)   # 30 days
    conversation_table.put_item(Item={
        "userID": user_id,
        "timestamp": now_in_seconds,
        "message": message,
        "role": role,
        "ttl": expire_ttl,
    })


# ---- Slide 32: read history (oldest first) ----------------------------------
def get_conversation_history(user_id):
    paginator = ddb_client.get_paginator("query")
    pages = paginator.paginate(
        TableName=TABLE_NAME,
        KeyConditionExpression="userID = :val",
        ExpressionAttributeValues={":val": {"S": user_id}},
    )
    messages = []
    for page in pages:
        for item in page.get("Items", []):
            messages.append({
                "role": item["role"]["S"],
                "content": [{"text": item["message"]["S"]}],
            })
    return messages
