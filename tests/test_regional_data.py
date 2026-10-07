"""
Tests for the regional-data connection. Run from the project root:
    python -m unittest discover -s tests -v

The CSV below is a TEST FIXTURE written to a temp dir at test time, only to
exercise the loader/ranking logic. It is not agricultural data and is never
used by the app.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regional_data import RegionalDataStore, current_season, normalize_place  # noqa: E402
import recommendation_engine as eng  # noqa: E402
from datetime import date  # noqa: E402

HEADER = ("state,district,agri_year,season,soil_type,nitrogen_kg_ha,phosphorus_kg_ha,potassium_kg_ha,"
          "soil_ph,avg_temperature_c,avg_humidity_pct,seasonal_rainfall_mm,major_crop,secondary_crops\n")
ROWS = [
    # fixture-only values
    "Teststate,Alpha,2022,Kharif,Loamy,,,,6.5,28,80,700,Paddy,Arhar/Tur; Maize\n",
    "Teststate,Alpha,2023,Kharif,Loamy,,,,6.5,29,78,650,Paddy,Arhar/Tur; Maize\n",
    "Teststate,Alpha,2023,Rabi,Loamy,,,,6.5,20,60,90,Gram,Masoor\n",
    "Teststate,Beta,2023,Kharif,,,,,,27,82,800,Rice,\n",           # no soil, no secondary
    "Teststate,Gamma,2023,Kharif,,,,,,27,82,800,Maize,Rice\n",
    "Teststate,Delta,2023,Kharif,,,,,,27,82,800,Maize,Cotton\n",
    "Soilonly,Zeta,2023,Kharif,Clay,,,,7.0,27,82,800,,\n",         # soil but no crop labels
]


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "d.csv")
        with open(cls.path, "w") as f:
            f.write(HEADER + "".join(ROWS))
        cls.store = RegionalDataStore(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_normalize_and_season(self):
        self.assertEqual(normalize_place("Warangal District"), normalize_place("warangal"))
        self.assertEqual(current_season(date(2026, 7, 1)), "Kharif")
        self.assertEqual(current_season(date(2026, 1, 1)), "Rabi")
        self.assertEqual(current_season(date(2026, 4, 1)), "Zaid")

    def test_district_season_specific_with_aliases(self):
        r = self.store.lookup("Teststate", "Alpha District", "Kharif")
        self.assertEqual(r["level"], "district")
        self.assertEqual(r["crops"], ["Rice", "Pigeonpeas", "Maize"])      # Paddy->Rice, Arhar/Tur->Pigeonpeas
        self.assertEqual(r["other_season_crops"], ["Chickpea", "Lentil"])  # grown in Rabi only
        self.assertEqual(r["soil"]["ph"], 6.5)
        self.assertIsNone(r["soil"]["N"])                                   # missing stays None
        self.assertEqual(r["seasonal_climate"]["years_of_data"], 2)

    def test_state_fallback_needs_enough_districts(self):
        r = self.store.lookup("Teststate", "Unknown", "Kharif")
        self.assertEqual(r["level"], "state")
        self.assertEqual(set(r["crops"][:2]), {"Rice", "Maize"})  # tied at 5 each across the 4 districts
        self.assertIsNone(self.store.lookup("Soilonly", "Nowhere", "Kharif"))  # 1 district only -> no state view
        self.assertIsNone(self.store.lookup("Nostate", "Nowhere", "Kharif"))

    def test_soil_without_crops_is_kept(self):
        r = self.store.lookup("Soilonly", "Zeta", "Kharif")
        self.assertEqual(r["crops"], [])
        self.assertEqual(r["soil"]["ph"], 7.0)

    def test_missing_file_is_graceful(self):
        s = RegionalDataStore(os.path.join(self.tmp.name, "nope.csv"))
        self.assertFalse(s.available())
        self.assertIsNone(s.lookup("a", "b", "Kharif"))

    def test_malformed_file_is_graceful(self):
        bad = os.path.join(self.tmp.name, "bad.csv")
        open(bad, "w").write("foo,bar\n1,2\n")
        s = RegionalDataStore(bad)
        self.assertFalse(s.available())


class EngineTests(unittest.TestCase):
    ml = {"rice": 40.0, "maize": 10.0, "cotton": 5.0}
    crop_info = {k: {"name": k.capitalize(), "ideal_temperature": [20, 35],
                     "ideal_humidity": [50, 90], "ideal_rainfall": [50, 300]} for k in ml}
    weather = {"temperature": 25, "humidity": 70, "rainfall": 100}

    def test_weights_unchanged_without_soil(self):
        self.assertEqual(eng.effective_weights(False), eng.RECOMMENDATION_WEIGHTS)
        w = eng.effective_weights(True)
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_no_soil_data_means_original_scoring(self):
        recs, w = eng.build_recommendations(self.ml, ["Rice", "Maize"], self.crop_info, self.weather)
        self.assertNotIn("soil", w)
        self.assertTrue(all(r["score_breakdown"]["soil_fit"] is None for r in recs))

    def test_other_season_crop_scored_lower_and_explained(self):
        recs, _ = eng.build_recommendations(self.ml, ["Rice"], self.crop_info, self.weather,
                                            other_season_crops=["Cotton"])
        cotton = next(r for r in recs if r["crop_key"] == "cotton")
        self.assertEqual(cotton["score_breakdown"]["regional_popularity"], eng.OTHER_SEASON_REGIONAL_SCORE)
        self.assertTrue(any("different season" in x["text"] for x in cotton["reasons"]))

    def test_soil_signal_activates(self):
        recs, w = eng.build_recommendations(self.ml, ["Rice"], self.crop_info, self.weather,
                                            regional_soil={"N": None, "P": None, "K": None, "ph": 6.5})
        self.assertIn("soil", w)
        rice = next(r for r in recs if r["crop_key"] == "rice")
        self.assertIsNotNone(rice["score_breakdown"]["soil_fit"])


if __name__ == "__main__":
    unittest.main()
