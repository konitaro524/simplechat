# lambda/index.py
import json
import os
import boto3
import re
from botocore.exceptions import ClientError


# ──────────────────────────────────────────────────────────────────────────────
#  ユーティリティ
# ──────────────────────────────────────────────────────────────────────────────
def extract_region_from_arn(arn: str) -> str:
    """Lambda ARN からリージョン名を取り出す"""
    match = re.search(r'arn:aws:lambda:([^:]+):', arn)
    return match.group(1) if match else "us-east-1"


# ──────────────────────────────────────────────────────────────────────────────
#  環境変数／グローバル
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID   = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")
# 例: "us.amazon.nova-lite-v1:0" → "nova-lite-v1"
MODEL_NAME = MODEL_ID.split(":")[0].split(".")[-1]

bedrock_client = None   # 再利用用のシングルトン


# ──────────────────────────────────────────────────────────────────────────────
#  Lambda ハンドラ
# ──────────────────────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    global bedrock_client

    try:
        # 1) Bedrock クライアント初期化
        if bedrock_client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            bedrock_client = boto3.client("bedrock-runtime", region_name=region)
            print(f"[Init] Bedrock client initialized in region: {region}")

        print(f"[Event] {json.dumps(event)[:400]}...")  # 長すぎる場合は一部のみ出力

        # 2) Cognito ユーザーを取得（任意）
        user_info = (
            event.get("requestContext", {})
                 .get("authorizer", {})
                 .get("claims", {})
        )
        if user_info:
            print(f"[Auth] user={user_info.get('email') or user_info.get('cognito:username')}")

        # 3) リクエストボディ解析
        body                = json.loads(event["body"])
        user_message        = body["message"]
        conversation_history = body.get("conversationHistory", [])

        print(f"[Input] msg='{user_message}' | model_id='{MODEL_ID}'")

        # 4) 会話履歴構築（最初にモデル名の SYSTEM メッセージを追加）
        messages = [{
            "role":    "system",
            "content": f"You are chatting with model **{MODEL_NAME}**."
        }] + conversation_history

        messages.append({"role": "user", "content": user_message})

        # Bedrock 形式に変換
        bedrock_messages = [
            {
                "role"   : m["role"],
                "content": [{"text": m["content"]}]
            } for m in messages
        ]

        # 5) Bedrock 推論呼び出し
        payload = {
            "messages"       : bedrock_messages,
            "inferenceConfig": {
                "maxTokens"   : 512,
                "stopSequences": [],
                "temperature" : 0.7,
                "topP"        : 0.9
            }
        }

        print(f"[Invoke] payload={json.dumps(payload)[:400]}...")

        response = bedrock_client.invoke_model(
            modelId    = MODEL_ID,
            body       = json.dumps(payload),
            contentType= "application/json"
        )

        res_body = json.loads(response["body"].read())
        print(f"[Bedrock] raw_response={json.dumps(res_body)[:400]}...")

        assistant_response = (
            res_body["output"]["message"]["content"][0]["text"]
        )

        messages.append({"role": "assistant", "content": assistant_response})

        # 6) 成功レスポンス
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type"                 : "application/json",
                "Access-Control-Allow-Origin"  : "*",
                "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods" : "OPTIONS,POST"
            },
            "body": json.dumps({
                "success"          : True,
                "modelId"          : MODEL_ID,      # ←★ クライアントにモデルIDを返却
                "response"         : assistant_response,
                "conversationHistory": messages
            })
        }

    except Exception as e:
        print(f"[Error] {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type"                 : "application/json",
                "Access-Control-Allow-Origin"  : "*",
                "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods" : "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error"  : str(e)
            })
        }
