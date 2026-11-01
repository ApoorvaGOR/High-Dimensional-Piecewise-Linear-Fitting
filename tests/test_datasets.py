"""Integration invariants for the pinned source workbooks, when cached locally."""
from pathlib import Path
import unittest
import numpy as np
from datasets import ROOT, SOURCES, acquire, load_dataset, sha256


@unittest.skipUnless(all((ROOT/'data/raw'/s['filename']).exists() for s in SOURCES.values()),
                     'Pinned workbooks are not cached; run python datasets.py to enable dataset checks')
class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data={name:load_dataset(name,offline=True) for name in SOURCES}

    def test_pinned_checksums(self):
        for name,spec in SOURCES.items():
            self.assertEqual(sha256(acquire(name,offline=True)),spec['sha256'])

    def test_era5_all_locations_and_requested_years(self):
        d=self.data['era5']
        self.assertEqual(d.Y.shape,(40401,26))
        np.testing.assert_array_equal(d.x,np.arange(2000,2026))
        xy=np.asarray(d.metadata['coordinates'])
        self.assertEqual(len(np.unique(xy[:,0])),201)
        self.assertEqual(len(np.unique(xy[:,1])),201)
        self.assertEqual(len({tuple(c) for c in xy}),40401)

    def test_holoassist_full_horizon_and_repeats(self):
        d=self.data['holoassist']
        self.assertEqual(d.Y.shape,(1092,240))
        self.assertEqual(len(np.unique(d.x)),232)
        self.assertEqual(sum(n.startswith('Left_hand:') for n in d.names),546)
        self.assertEqual(sum(n.startswith('Right_hand:') for n in d.names),546)
        self.assertAlmostEqual(d.x[0],.0418699,places=7)
        self.assertAlmostEqual(d.x[-1],8.0112456,places=7)

    def test_cmapss_engine1_sensors_and_rul_axis(self):
        d=self.data['cmapss']
        self.assertEqual(d.Y.shape,(21,192))
        np.testing.assert_array_equal(d.x,np.arange(1,193))
        self.assertEqual(d.metadata['initial_RUL'],191)
        self.assertTrue(all(n.startswith('Sensor ') for n in d.names))
        self.assertEqual(d.metadata['constant_dimensions'],7)


if __name__=='__main__':
    unittest.main()
