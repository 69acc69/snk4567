import unittest
from database.timescaledb_interface import TimescaleDBInterface
from feature_engineering.feature_engineer import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.db = TimescaleDBInterface()
        self.feature_engineer = FeatureEngineer(self.db)
    
    def test_compute_features(self):
        try:
            self.feature_engineer.compute_features()
            df = self.db.get_features()
            self.assertIn('vwap_30m', df.columns)
            self.assertIn('engulfing', df.columns)
            self.assertIn('volume_profile', df.columns)
            self.assertIn('pullback_to_vwap', df.columns)
            self.assertIn('higher_high', df.columns)
            self.logger.info("FeatureEngineer unit test passed.")
        except Exception as e:
            self.fail(f"compute_features() raised Exception {e}")

if __name__ == '__main__':
    unittest.main()
