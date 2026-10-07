"""Run from project root: python -m unittest discover -s tests -v"""
import os, sys, unittest, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import joblib, pandas as pd  # noqa: E402
from explain import explain_prediction  # noqa: E402


class ExplainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(os.path.join("models", "crop_model.pkl"))
        cls.prep = joblib.load(os.path.join("models", "preprocessing.pkl"))
        cls.names = cls.prep["feature_names"]
        cls.raw = {"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82, "ph": 6.5, "rainfall": 200}
        cls.scaled = cls.prep["scaler"].transform(pd.DataFrame([[cls.raw[f] for f in cls.names]], columns=cls.names))

    def test_shap_is_additive_and_covers_all_features(self):
        ex = explain_prediction(self.model, self.prep["label_encoder"], self.names, self.raw, self.scaled, ["rice"])["rice"]
        self.assertEqual(ex["method"], "shap")
        self.assertEqual({f["feature"] for f in ex["features"]}, set(self.names))
        prob = self.model.predict_proba(self.scaled)[0][list(self.prep["label_encoder"].classes_).index("rice")] * 100
        self.assertAlmostEqual(ex["base_value"] + sum(f["contribution"] for f in ex["features"]), prob, delta=0.2)
        self.assertIn("not proof", ex["note"])

    def test_unknown_crop_skipped(self):
        self.assertEqual(explain_prediction(self.model, self.prep["label_encoder"], self.names, self.raw, self.scaled, ["wheat"]), {})

    def test_fallback_to_global_importance(self):
        import explain
        explain._explainer, explain._explainer_failed = None, True
        try:
            ex = explain_prediction(self.model, self.prep["label_encoder"], self.names, self.raw, self.scaled, ["rice"])["rice"]
            self.assertEqual(ex["method"], "feature_importance")
        finally:
            explain._explainer_failed = False


if __name__ == "__main__":
    unittest.main()
