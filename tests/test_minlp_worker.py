"""Check that an isolated worker reports failures without crashing its caller."""
from pathlib import Path
import tempfile
import unittest
from minlp_worker import run_isolated


class WorkerTests(unittest.TestCase):
    def test_missing_input_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(RuntimeError,'FileNotFoundError'):
                run_isolated(root/'absent.npz',root/'bounds.npz',root/'result.json',
                             root/'solution.npz',root/'solver.log',k=1,time_limit=1)
            self.assertTrue((root/'result.json').exists())


if __name__=='__main__':
    unittest.main()
