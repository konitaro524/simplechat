import json, os, re, urllib.request, ssl

# FastAPI エンドポイントを環境変数か直書きで指定
API_URL = os.environ.get(
    "FASTAPI_ENDPOINT",
    "https://7d30-34-19-104-59.ngrok-free.app/generate"   # ★←自分の URL
)

# ngrok の自己署名証明書を許可
ssl._create_default_https_context = ssl._create_unverified_context


def extract_region_from_arn(arn: str) -> str:
    m = re.search(r"arn:aws:lambda:([^:]+):", arn)
    return m.group(1) if m else "us-east-1"


def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # Body からメッセージを取得
        body = json.loads(event["body"])
        message = body["message"]
        conversation_history = body.get("conversationHistory", [])
        print("Processing message:", message)

        # FastAPI へ POST するペイロードを作成
        req_payload = json.dumps(
            {
                "prompt": message,           # 単純化：会話履歴は付けず prompt のみ
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            API_URL,
            data=req_payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=30) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))

        print("FastAPI response:", resp_body)
        assistant_response = resp_body["generated_text"]

        # 会話履歴を更新（UI 側で使うなら）
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": assistant_response})

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "success": True,
                    "response": assistant_response,
                    "conversationHistory": conversation_history,
                }
            ),
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"success": False, "error": str(e)}),
        }
