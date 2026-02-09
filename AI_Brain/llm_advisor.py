from google import genai
import logging
import json
import re

logger = logging.getLogger("LLMRewardAdvisor")

class LLMRewardAdvisor:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        if api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                self.cooldown_until = 0 # Unix timestamp
                logger.info(f"LLM Reward Advisor initialized with {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Client: {e}")

    async def get_quality_score(self, trade_data):
        """
        Calculates a trade quality score (0-100).
        Tier 1: Gemini 1.5
        Tier 2: Rule-Based Fallback
        """
        if self.client:
            import time
            if time.time() < self.cooldown_until:
                logger.warning("LLM Advisor in cooldown. Using rule-based fallback.")
                return self._get_rule_based_score(trade_data)

            try:
                import asyncio
                # Simple retry logic for 429
                for attempt in range(2):
                    try:
                        return await self._get_llm_score(trade_data)
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            import time
                            self.cooldown_until = time.time() + 60 # 60s cooldown
                            logger.warning(f"LLM Rate Limited (429). Attempt {attempt+1} failed. Cooling down.")
                            if attempt == 0:
                                await asyncio.sleep(2) # Short wait before retry
                                continue
                        raise e
            except Exception as e:
                logger.warning(f"LLM Assessment failed after retries. Error: {e}")
        
        return self._get_rule_based_score(trade_data)

    async def _get_llm_score(self, trade_data):
        prompt = f"""
        You are a Senior Quantitative Trading Mentor. Your task is to evaluate the QUALITY of a trade execution.
        
        TRADE DATA:
        - Symbol: {trade_data.get('symbol')}
        - Action: {trade_data.get('action')}
        - Pattern Detected: {trade_data.get('pattern_name')}
        - Entry Price: {trade_data.get('open_price')}
        - Exit Price: {trade_data.get('close_price')}
        - Net PnL: ${trade_data.get('net_profit')}
        - CNN Confidence: {trade_data.get('cnn_confidence', 0)}
        
        SCORING CRITERIA:
        1. Professionalism: Did the entry align with the detected pattern?
        2. Risk/Reward: Is the win based on a good move or just noise?
        3. Discipline: High confidence entries should be rewarded more if they win.
        
        INSTRUCTION:
        Assign a Decision Quality Score from 0 to 100. 
        - 80-100: Perfect execution following technical patterns.
        - 50-79: Acceptable trade but could be improved.
        - 0-49: Poor execution, lucky win, or gambling against trend.
        
        Return ONLY a JSON object:
        {{"score": int, "reasoning": "short string in Thai"}}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Clean response for JSON parsing
            clean_text = re.search(r'\{.*\}', response.text, re.DOTALL)
            if clean_text:
                result = json.loads(clean_text.group())
                return int(result.get('score', 50)), result.get('reasoning', "ไม่มีคำอธิบาย")
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Gemini processing error: {err_msg}")
            raise e
            
        return 50, "LLM parsing failed"

    def _get_rule_based_score(self, trade_data):
        """
        Level 3 Fallback: Purely mathematical scoring based on confidence and PnL.
        """
        logger.info("Using Level 3: Rule-Based Scoring Fallback.")
        pnl = trade_data.get('net_profit', 0)
        conf = trade_data.get('cnn_confidence', 0.5)
        
        # Base score starts at 50
        score = 50
        
        # Reward high confidence wins
        if pnl > 0:
            score += (conf * 30) # Max +30
        else:
            # Penalize high confidence losses (overconfidence)
            score -= (conf * 30) # Max -30
            
        # Clamp between 0-100
        score = max(0, min(100, score))
        return int(score), "คำนวณโดยระบบสำรอง (Rule-based) เนื่องจาก LLM ไม่พร้อมใช้งาน"


