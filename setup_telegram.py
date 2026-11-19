import json
 
import requests
 
token = input("Paste your bot token (from BotFather): ").strip().strip("<>").strip()
if token.lower().startswith("bot"):
    token = token[3:]
 
base = f"https://api.telegram.org/bot{token}"
try:
    data = requests.get(base + "/getUpdates", timeout=15).json()
except Exception as e:
    raise SystemExit(f"Could not reach Telegram: {e}")
 
if not data.get("ok"):
    raise SystemExit(f"Token problem: {data}. Copy the full token again from BotFather.")
 
chat_id = None
for u in reversed(data["result"]):
    msg = u.get("message") or u.get("edited_message")
    if msg:
        chat_id = msg["chat"]["id"]
        break
 
if chat_id is None:
    raise SystemExit("Token is OK, but no message found. Open YOUR bot in Telegram, "
                     "press Start, send 'hi', then run this again.")
 
with open("telegram.json", "w") as f:
    json.dump({"token": token, "chat_id": str(chat_id)}, f)
 
r = requests.post(base + "/sendMessage",
                  data={"chat_id": chat_id, "text": "Bed monitor connected successfully."},
                  timeout=15)
print("telegram.json created. Test message sent." if r.ok else f"Saved, but send failed: {r.text}")
 

