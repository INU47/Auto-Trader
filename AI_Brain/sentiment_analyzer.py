import httpx
import logging
import json
import asyncio

logger = logging.getLogger("SentimentAnalyzer")

class SentimentAnalyzer:
    def __init__(self, api_key="sk-1i6oPb2GJa3uYhbcMNf599IPTzFEJyy6UumyLH6EIWjS9adq"):
        self.api_key = api_key
        self.base_url = "https://api.opentyphoon.ai/v1/chat/completions"
        self.timeout = httpx.Timeout(20.0, connect=5.0)

    async def analyze_news_sentiment(self, news_text):
        """
        Analyzes news text and returns a sentiment score from -1.0 to 1.0.
        -1.0: Extremely Bearish
         0.0: Neutral
         1.0: Extremely Bullish
        """
        if not news_text:
            return 0.0

        prompt = f"""
        Analyze the following financial news and provide a sentiment score for the market.
        Respond ONLY with a JSON object containing the float value 'sentiment' between -1.0 and 1.0.
        
        News: {news_text}
        
        JSON Result:
        """
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "typhoon-v1.5-instruct",
                        "messages": [
                            {"role": "system", "content": "You are a specialized financial analyst. Respond only in JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 50
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content'].strip()
                    import re
                    match = re.search(r"({.*})", content, re.DOTALL)
                    if match:
                        result = json.loads(match.group(1))
                        return float(result.get('sentiment', 0.0))
                else:
                    logger.error(f"Typhoon API Error: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"Sentiment Analysis Failed: {e}")
            
        return 0.0

    async def get_latest_market_sentiment(self, symbol="EURUSD"):
        """
        Fetches latest news from Finnhub and analyzes sentiment via Typhoon.
        """
        finnhub_key = "d6i0349r01qr5k4dhmh0d6i0349r01qr5k4dhmhg"
        base_pair = symbol[:3] # EUR, GBP, XAU, etc.
        
        # Finnhub News API
        url = f"https://finnhub.io/api/v1/news?category=forex&token={finnhub_key}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    news_list = response.json()
                    # Filter news related to the symbol or category
                    relevant_text = ""
                    for item in news_list[:5]: # Take top 5 latest
                        relevant_text += item.get('headline', '') + ". " + item.get('summary', '') + " "
                    
                    if relevant_text:
                        return await self.analyze_news_sentiment(relevant_text)
                else:
                    logger.error(f"Finnhub API Error: {response.status_code}")
        except Exception as e:
            logger.error(f"Finnhub Fetch Failed: {e}")
            
        return 0.0
