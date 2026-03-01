from google import genai
import logging
import json
import re
import asyncio

logger = logging.getLogger("LLMRewardAdvisor")

class LLMRewardAdvisor:
    def __init__(self, api_key_pool=None, model_pool=None, max_cooldown=300, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            logger.info("⚡ LLM Disabled by config. Using Rule-Based Fallback only.")
            return

        self.api_key_pool = api_key_pool if isinstance(api_key_pool, list) else [api_key_pool]
        self.current_key_idx = 0
        
        self.model_pool = model_pool or ["gemini-1.5-flash"]
        self.active_pool = list(self.model_pool) 
        self.current_model_idx = 0
        
        self.max_cooldown = max_cooldown
        self.client = None
        self.cooldown_until = 0 
        self.consecutive_429s = 0
        self.is_connected = False
        self.blacklisted_models = set()
        
        self._initialize_client()

    def _initialize_client(self):
        if not self.api_key_pool:
            logger.error("No API Keys provided!")
            self.is_connected = False
            return

        current_key = self.api_key_pool[self.current_key_idx]
        try:
            masked_key = f"{current_key[:5]}...{current_key[-4:]}"
            logger.info(f"🔑 Initializing Gemini with Key #{self.current_key_idx + 1} ({masked_key})")
            
            self.client = genai.Client(api_key=current_key)
            self.is_connected = True
            
            self.active_pool = list(self.model_pool)
            self.blacklisted_models.clear()
            self.current_model_idx = 0
            
            logger.info(f"✅ LLM Ready. Pool: {len(self.active_pool)} models.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Client: {e}")
            self.is_connected = False
            self._rotate_key()

    def _rotate_key(self):
        if len(self.api_key_pool) <= 1:
            logger.warning("No other API keys to rotate to.")
            return False
            
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_pool)
        logger.warning(f"🔄 Rotating to API Key #{self.current_key_idx + 1}...")
        self._initialize_client()
        return True

    @property
    def is_healthy(self):
        if not self.client or not self.is_connected:
            return False
        import time
        now = time.time()
        if now < self.cooldown_until:
            return False
            
        if self.consecutive_429s >= 5:
            return False
        return True

    async def get_quality_score(self, trade_data):

        if not self.enabled:
            return self._get_rule_based_score(trade_data)

        if not self.client or not self.is_connected:
             self._initialize_client()

        import asyncio
        import time

        while True:
            keys_tried = 0
            max_keys = len(self.api_key_pool)
            
            while keys_tried < max_keys:
                now = time.time()
                if now < self.cooldown_until:
                     remaining = self.cooldown_until - now
                     if remaining > 0:
                         logger.info(f"⏳ Still in cooldown. Waiting {int(remaining)}s...")
                         await asyncio.sleep(remaining)

                try:
                    success = False
                    for _ in range(len(self.active_pool)):
                        if not self.active_pool: break
                        
                        model_name = self.active_pool[self.current_model_idx]
                        try:
                            score, reason = await self._get_llm_score(trade_data, model_name)
                            self.consecutive_429s = 0
                            return score, reason
                        except Exception as e:
                            err_str = str(e)
                            
                            if "limit: 0" in err_str or "404" in err_str or "NOT_FOUND" in err_str:
                                logger.error(f"🚫 Model {model_name} unusable ({err_str[:50]}). Blacklisting.")
                                if model_name in self.active_pool:
                                    self.active_pool.remove(model_name)
                                if self.active_pool:
                                    self.current_model_idx %= len(self.active_pool)
                                continue 
                                
                            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                                if self.active_pool:
                                    self.current_model_idx = (self.current_model_idx + 1) % len(self.active_pool)
                                logger.warning(f"LLM Rate Limited ({model_name}). Trying next model...")
                                await asyncio.sleep(1)
                                continue 
                            
                            if "401" in err_str or "API_KEY_INVALID" in err_str:
                                self.is_connected = False 
                                logger.error(f"Critical API Error: Invalid API Key #{self.current_key_idx + 1}")
                                break
                            
                            raise e
                    
                    keys_tried += 1
                    if keys_tried < max_keys:
                        logger.warning(f"⚠️ Key #{self.current_key_idx + 1} exhausted. Rotating...")
                        self._rotate_key()
                    else:
                        break
                        
                except Exception as e:
                    logger.warning(f"Unexpected error with Key #{self.current_key_idx + 1}: {e}")
                    keys_tried += 1
                    self._rotate_key()

            self.consecutive_429s += 1
            retry_minutes = 30 
            wait_time = retry_minutes * 60
            
            self.cooldown_until = time.ctime(time.time() + wait_time)
            logger.warning(f"🛑 ALL {max_keys} KEYS EXHAUSTED. Waiting {retry_minutes} mins for quota recovery...")
            logger.warning(f"⏳ Sleeping until {self.cooldown_until}...")
            
            self.cooldown_until = time.time() + wait_time
            await asyncio.sleep(wait_time)
            
            self.cooldown_until = 0
            self.consecutive_429s = 0
            self.active_pool = list(self.model_pool)
            logger.info("♻️ Waking up from quota sleep. Restarting full rotation cycle...")

    async def _get_llm_score(self, trade_data, model_name):
        prompt = f"""
        You are an AI Trading Mentor expert in algorithmic trading analysis.
        Your task is to evaluate the quality of a recently closed trade order to provide reinforcement for our AI models.
        
        Trade Data:
        - Symbol: {trade_data.get('symbol')}
        - Action: {trade_data.get('action')}
        - Open Time: {trade_data.get('open_time')}
        - Close Time: {trade_data.get('close_time')}
        - CNN Pattern Confidence: {trade_data.get('cnn_confidence', 0.5):.2f}
        - LSTM Trend Confidence: {trade_data.get('lstm_confidence', trade_data.get('lstm_trend_pred', 0.5)):.2f}
        - Net Profit: ${trade_data.get('net_profit', 0):.2f}
        - Duration: {trade_data.get('duration_minutes', 0)} mins
        
        Scoring Rules (Score 0-100):
        1. High Score (70-100): Profitable trades with high confidence, or small losses where AI followed the trend correctly.
        2. Low Score (0-30): Large losses from counter-trend entries, or cases of extreme overconfidence.
        3. 50 Score: Standard/Neutral quality.
        
        You MUST respond in valid JSON format only, with this structure:
        {{
            "score": <number 0-100>,
            "reasoning": "<short summary explanation in English, 1-2 sentences>"
        }}
        
        IMPORTANT: Do NOT include any other text except for the JSON. No ```json blocks or introductory text. Respond ONLY with JSON.
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model_name,
                contents=prompt
            )
            
            text = response.text.strip()
            
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            
            clean_text = json_match.group(1) if json_match else text
            
            try:
                result = json.loads(clean_text)
                score = int(result.get('score', 50))
                reason = result.get('reasoning', "No description")
                return score, reason
            except json.JSONDecodeError:
                score_match = re.search(r'"score":\s*(\d+)', text)
                reason_match = re.search(r'"reasoning":\s*"(.*?)"', text)
                
                score = int(score_match.group(1)) if score_match else 50
                reason = reason_match.group(1) if reason_match else "LLM parsing failed (Fallback)"
                
                if score_match or reason_match:
                    return score, reason
                
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Gemini API call failed ({model_name}): {err_msg[:100]}...")
            raise e
            
        return 50, "LLM parsing failed"

    def _get_rule_based_score(self, trade_data):
        logger.info("Using Level 3: Rule-Based Scoring Fallback.")
        pnl = trade_data.get('net_profit', 0)
        conf = trade_data.get('cnn_confidence', 0.5)
        
        score = 50
        
        if pnl > 0:
            score += (conf * 30)
        else:
            score -= (conf * 30)
            
        score = max(0, min(100, score))
        return int(score), "Calculated by fallback system (Rule-based) as LLM is unavailable"


