import pytest
from ml.training.data_validation import validate_schema

class TestDataValidation:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            "timestamp": ["2023-01-01 12:00:00"],
            "src_bytes": [1024],
            "is_malicious": [False]
        })

    def test_missing_columns(self, sample_data):
        with pytest.raises(ValueError, match="Missing required column: dst_bytes"):
            validate_schema(sample_data)

    def test_data_types(self):
        invalid_data = sample_data.copy()
        invalid_data["src_bytes"] = "not_an_int"
        with pytest.raises(TypeError, match="Invalid type for src_bytes"):
            validate_schema(invalid_data)