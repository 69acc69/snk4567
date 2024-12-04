import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    # Bybit API Credentials
    API_KEY = os.getenv('BYBIT_API_KEY')
    SECRET = os.getenv('BYBIT_SECRET')

    # Database Configuration
    DB_CONFIG = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT', 5432))
    }

    RISK_PER_TRADE = 0.01
    CURRENCY = 'ETH'
    SYMBOL = 'ETHUSD'
    TIMEFRAME = '15m'
    
    # Twitter API Credentials
    TWITTER_API_KEY = 'your_twitter_api_key'
    TWITTER_API_SECRET = 'your_twitter_api_secret'
    TWITTER_ACCESS_TOKEN = 'your_twitter_access_token'
    TWITTER_ACCESS_SECRET = 'your_twitter_access_secret'

    # Telegram Bot Credentials
    TELEGRAM_BOT_TOKEN = 'your_telegram_bot_token'
    TELEGRAM_CHAT_ID = 'your_telegram_chat_id'