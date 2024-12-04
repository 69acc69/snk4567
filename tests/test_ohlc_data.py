import unittest
from data_collection.ohlc_data import OHLCDataCollector
from utils.config import Config

class TestOHLCDataCollector(unittest.TestCase):
    def setUp(self):
        self.collector = OHLCDataCollector(Config.API_KEY, Config.SECRET)
    
    def test_fetch_historical_ohlc(self):
        result = self.collector.fetch_historical_ohlc('ETHUSD', '1m', 1)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
