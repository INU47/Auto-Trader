import httpx
import logging
import json
import os

logger = logging.getLogger("TelegramNotifier")

class TelegramNotifier:
    def __init__(self, config_path="Config/server_config.json"):
        self.enabled = False
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.token = config.get('telegram_token')
                self.chat_id = config.get('chat_id')
                if self.token and self.chat_id and "YOUR_BOT_TOKEN" not in self.token:
                    self.enabled = True
                    self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
                else:
                    logger.warning("Telegram Bot Token not configured properly. Alerts disabled.")
            
            self.offset = 0
        except Exception as e:
            logger.error(f"Failed to load Telegram config: {e}")

    async def send_message(self, text, reply_markup=None):
        if not self.enabled:
            logger.info(f"TELEGRAM (Disabled): {text}")
            return
            
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": f"🤖 **QuantSystem**\n{text}",
                "parse_mode": "Markdown"
            }
            if reply_markup:
                payload["reply_markup"] = reply_markup
                
            async with httpx.AsyncClient() as client:
                resp = await client.post(self.api_url, json=payload, timeout=10.0)
                if resp.status_code == 400 and "can't parse entities" in resp.text:
                    logger.warning("Telegram Markdown failure, retrying as plain text...")
                    payload["text"] = f"🤖 QuantSystem\n{text}"
                    del payload["parse_mode"]
                    resp = await client.post(self.api_url, json=payload, timeout=10.0)
                
                if resp.status_code != 200:
                    logger.error(f"Telegram API Error: {resp.text}")
        except (httpx.ConnectTimeout, httpx.ConnectError):
            logger.warning(f"Telegram connection timed out: {text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def get_main_menu(self):
        return {
            "keyboard": [
                [{"text": "/info"}, {"text": "/history"}],
                [{"text": "/on"}, {"text": "/off"}]
            ],
            "resize_keyboard": True,
            "one_time_keyboard": False
        }

    async def check_commands(self):
        if not self.enabled: return []
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {"offset": self.offset, "limit": 10, "timeout": 1}
            
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params, timeout=5.0)
                resp.raise_for_status()
                if resp.status_code == 200:
                    data = resp.json()
                    new_cmds = []
                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            self.offset = update["update_id"] + 1
                            message = update.get("message", {})
                            text = message.get("text", "")
                            from_id = str(message.get("from", {}).get("id", ""))
                            chat_id_msg = str(message.get("chat", {}).get("id", ""))
                            expected_chat_id = str(self.chat_id)
                            
                            logger.info(f"📨 Update received: from_id={from_id}, chat_id={chat_id_msg}, expected={expected_chat_id}, text='{text}'")
                            
                            if (from_id == expected_chat_id or chat_id_msg == expected_chat_id) and text:
                                new_cmds.append(text.lower().strip())
                                logger.info(f"✅ Command accepted: {text}")
                            else:
                                logger.warning(f"⚠️ Command rejected: from_id={from_id}, chat_id={chat_id_msg} != expected={expected_chat_id}")
                    return new_cmds
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError):
            pass
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                logger.error("Telegram Conflict (409): Another instance is likely running. Close other terminal windows.")
            else:
                logger.error(f"Telegram HTTP Error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.error(f"Telegram polling error ({type(e).__name__}): {e}")
        return []
