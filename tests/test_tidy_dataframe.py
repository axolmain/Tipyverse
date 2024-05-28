import unittest
import polars as pl
from src.tipyverse.tidy_dataframe import TidyDataFrame


class TestTidyDataFrame(unittest.TestCase):

    def setUp(self):
        self.df = TidyDataFrame({
            "species": ["Adelie", "Chinstrap", "Gentoo", "Adelie"],
            "island": ["Torgersen", "Biscoe", "Dream", "Dream"],
            "body_mass_g": [3750, 3800, 5000, 3700]
        })

    def test_select(self):
        result = self.df.select("species", "body_mass_g")
        self.assertEqual(list(result.columns), ["species", "body_mass_g"])

    def test_filter(self):
        result = self.df.filter(species="Adelie")
        self.assertEqual(result.shape[0], 2)

    def test_arrange(self):
        result = self.df.arrange("body_mass_g", ascending=False)
        self.assertEqual(result[0, "body_mass_g"], 5000)

    def test_mutate(self):
        result = self.df.mutate(body_mass_kg=pl.col("body_mass_g") / 1000)
        self.assertIn("body_mass_kg", result.columns)
        self.assertAlmostEqual(result[0, "body_mass_kg"], 3.75, places=2)

    def test_group_by_and_summarize(self):
        # Perform grouping and summarization
        result = (self.df
                  .group_by("species")
                  .summarize(total_mass="body_mass_g"))

        # Expected values: sum of body masses per species
        expected_results = {
            "Adelie": 7450,  # Sum of 3750 and 3700
            "Chinstrap": 3800,
            "Gentoo": 5000
        }

        # Verify the shape: 3 species
        self.assertEqual(result.shape[0], 3)

        # Verify each group's aggregation
        for index, row in enumerate(result.to_pandas().itertuples(index=False)):
            self.assertAlmostEqual(row.total_mass, expected_results[row.species], places=0)


if __name__ == "__main__":
    unittest.main()
