import pytest
from ml.preprocessing import DataPreprocessor
import pandas as pd
import numpy as np

class TestDataPreprocessor:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            "timestamp": ["2023-01-01 12:00:00"],
            "src_ip": ["192.168.1.1"],
            "protocol": ["tcp"],
            "payload_length": [1024],
            "is_malicious": [0]
        })

    def test_ip_anonymization(self, sample_data):
        preprocessor = DataPreprocessor()
        processed = preprocessor.transform(sample_data)
        assert not processed["src_ip"].str.contains(r"\d+\.\d+\.\d+\.\d+").any()

    def test_missing_data_handling(self):
        preprocessor = DataPreprocessor()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocessor.transform(pd.DataFrame())