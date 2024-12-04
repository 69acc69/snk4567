from apscheduler.schedulers.background import BackgroundScheduler
from utils.config import Config
from utils.logger import get_logger
import psutil
import requests
import ccxt
from ai_models.ai_model import AIModel
from feature_engineering.feature_engineer import FeatureEngineer

class SystemMonitor:
    """
    Class for monitoring the health of the trading system.
    """
    def __init__(self, db_interface):
        self.scheduler = BackgroundScheduler()
        self.logger = get_logger(self.__class__.__name__)
        self.exchange = ccxt.bitfinex({
            'apiKey': Config.API_KEY,
            'secret': Config.SECRET,
        })
        self.ai_model = AIModel(db_interface)
        self.ai_model.load_model()
        self.feature_engineer = FeatureEngineer(db_interface)
        self.latest_features = None

    def start(self):
        """
        Start the system monitor.
        """
        self.scheduler.add_job(self.health_check, 'interval', minutes=1)
        self.scheduler.start()
        self.logger.info("System Monitor started.")

    def health_check(self):
        """
        Perform health checks on system resources and components.
        """
        # CPU and Memory Usage
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        if cpu_usage > 80:
            self.logger.warning(f"High CPU usage: {cpu_usage}%")
        if memory_usage > 80:
            self.logger.warning(f"High memory usage: {memory_usage}%")

        # Check Trade Execution
        self.check_trade_execution()

        # Check Model Health
        self.check_model_health()

        # Alert System
        if cpu_usage > 90 or memory_usage > 90:
            self.send_alert(f"System resources critically high. CPU: {cpu_usage}%, Memory: {memory_usage}%")

    def check_trade_execution(self):
        """
        Verify that trades are being executed as expected.
        """
        try:
            open_orders = self.exchange.fetch_open_orders()
            if open_orders:
                self.logger.info(f"Open orders: {open_orders}")
            else:
                self.logger.info("No open orders.")
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}")
            self.send_alert(f"Error fetching open orders: {e}")

    def check_model_health(self):
        """
        Monitor the AI model's performance.
        """
        try:
            # Fetch latest features
            latest_features = self.feature_engineer.get_latest_features()
            # Prepare data for prediction
            X_latest = latest_features.drop(['timestamp']).values.reshape(1, -1)
            # Make prediction
            predictions = self.ai_model.xgb_model.predict(X_latest)
            # Evaluate model (implement your evaluation logic)
            accuracy = self.evaluate_model()
            if accuracy < 0.6:
                self.logger.warning(f"Model accuracy low: {accuracy}")
                self.send_alert(f"Model accuracy has dropped below threshold: {accuracy}")
        except Exception as e:
            self.logger.error(f"Error checking model health: {e}")
            self.send_alert(f"Error checking model health: {e}")

    def evaluate_model(self):
        """
        Evaluate the model's accuracy over recent data.
        Implement your evaluation logic here.
        """
        # Placeholder implementation
        return 1.0  # Assume perfect accuracy for placeholder

    def send_alert(self, message):
        """
        Send an alert via Telegram.
        """
        telegram_api_url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {
            'chat_id': Config.TELEGRAM_CHAT_ID,
            'text': message
        }
        try:
            response = requests.get(telegram_api_url, params=params)
            if response.status_code == 200:
                self.logger.info("Alert sent successfully.")
            else:
                self.logger.error(f"Failed to send alert: {response.text}")
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
