from google import genai
import logging
import json

logger = logging.getLogger("VirtualAnalyst")

class VirtualAnalyst:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        if not api_key:
            logger.error("Gemini API Key missing!")
            self.client = None
            return
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Virtual Analyst Initialized with model: {model_name}")

    async def generate_report(self, signal_data, mode="telegram"):
        """
        Generates a technical analysis report.
        mode: "telegram" (concise) or "dashboard" (trader log style)
        """
        if not self.client:
            return "⚠️ [Analyst Offline] Cannot generate analysis at the moment."

        if mode == "dashboard":
            prompt = f"""
            You are a "Senior Quant Trader" with extensive experience.
            Mission: Write a short trade log entry for a private Dashboard to serve as a memory aid.
            
            Data:
            - Symbol: {signal_data.get('symbol')}
            - Action: {signal_data.get('action')}
            - Pattern: {signal_data.get('pattern')}
            - Confidence: {int(signal_data.get('confidence', 0) * 100)}%
            - Outlook: {signal_data.get('future_outlook')}
            - Price: {signal_data.get('price')}

            Writing Style:
            - Write as if talking to yourself or taking mental notes (Think Aloud).
            - Use Trader terminology (Support, Resistance, Rejection, Volume).
            - Briefly analyze why this entry was taken (2-3 sentences).
            - Do not start with "Summary" or "Hello", just put the content.
            
            Example:
            "Found Bullish Engulfing at key M1 support. Price rejected beautifully with volume support. Confident in this one at 85%. Let's go!"
            """
        else:
            # Telegram Mode (Original)
            prompt = f"""
            You are a "Professional Quant Analyst".
            Mission: Summarize the AI analysis to be as concise (Glanceable) as possible for Telegram.
    
            Data:
            - Symbol: {signal_data.get('symbol')}
            - Guidance: {signal_data.get('action')}
            - Pattern: {signal_data.get('pattern')}
            - Confidence: {int(signal_data.get('confidence', 0) * 100)}%
            - Outlook: {signal_data.get('future_outlook')}
    
            Please write a 3-line summary as follows (Avoid special characters that break Markdown):
            🎯 Signal: [Action] [Symbol] ([Confidence]%)
            📊 Reason: [Short 1-sentence analysis]
            ⚠️ Risk: [Short risk advice]
    
            *Use a friendly but professional tone, be direct and concise.*
            """


        try:
            # google-genai SDK v2 syntax
            # Model name format: "models/gemini-1.5-flash" or just "gemini-1.5-flash"
            response = self.client.models.generate_content(
                model=f"models/{self.model_name}" if not self.model_name.startswith("models/") else self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            err_msg = str(e)
            
            # Check for specific error types
            if "404" in err_msg or "NOT_FOUND" in err_msg:
                logger.error(f"Model not found. Trying fallback model. Error: {e}")
                # Try with a different model format
                try:
                    response = self.client.models.generate_content(
                        model="gemini-1.5-flash-latest",
                        contents=prompt
                    )
                    return response.text
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    return "⚠️ [Analyst Error] AI Model unavailable. Please check API key and model name."
            
            if "429" in err_msg or "quota" in err_msg.lower():
                logger.warning("Gemini API Quota Exceeded. Analyst is silent.")
                return "⏸️ [Analyst Sleep] Free quota exceeded. System will resume automatically when quota resets."
            
            logger.error(f"Error generating LLM report: {e}")
            return f"⚠️ [Analyst Error] Analysis generation failed (Error: {err_msg[:50]}...)"
