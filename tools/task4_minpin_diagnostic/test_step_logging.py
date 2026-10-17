import json,sys,tempfile,unittest
from pathlib import Path
from run_logged_step import run_step
class TestLogging(unittest.TestCase):
    def test_exit_command_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            command=[sys.executable,'-c','print("synthetic-first-error"); raise SystemExit(7)']
            self.assertEqual(run_step(d,'failure',command),7)
            first=(Path(d)/'FIRST_FAILURE.json').read_bytes()
            record=json.loads(first);self.assertEqual(record['argv'],command);self.assertEqual(record['exit_code'],7)
            self.assertIn('synthetic-first-error',(Path(d)/'failure.log').read_text())
            with self.assertRaises(FileExistsError):run_step(d,'failure',command)
            self.assertEqual(run_step(d,'dependency_missing',['/no/such/executable_task4']),127)
            self.assertEqual((Path(d)/'FIRST_FAILURE.json').read_bytes(),first)
            self.assertEqual(run_step(d,'pass',[sys.executable,'-c','pass']),0)
if __name__=='__main__':unittest.main()
