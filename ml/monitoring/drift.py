import pandas as pd
from scipy.stats import ks_2samp
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DriftDetector:

    def __init__(self):
        self.reference = self._load_reference()




    def _load_reference(self):

        try:
            with open('ml/models/deployment_packages/latest/reference_stats.json') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Reference stats not found. Generate during training!")
            return None




    def check_drift(self, features: dict):

        if not self.reference:
            return None

        drift_results = {}

        for feat_name, value in features.items():
            if feat_name in self.reference['features']:
                stat, pval = ks_2samp([value], self.reference['features'][feat_name]['samples'])
                drift_results[feat_name] = {
                    'ks_statistic': stat,
                    'p_value': pval,
                    'drift_detected': pval < 0.05
                }

        self._log_drift(drift_results)
        return drift_results




    def _log_drift(self, results: dict):

        log_file = Path("ml/monitoring/drift_logs.csv")
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'drift_features': [k for k, v in results.items() if v['drift_detected']]
        }

        log_file.parent.mkdir(exist_ok=True)
        pd.DataFrame([log_entry]).to_csv(
            log_file,
            mode='a',
            header=not log_file.exists(),
            index=False
        )
