import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from preflight_cpu import audit

class TestPreflight(unittest.TestCase):
    def test_exact_artifact_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def git(*args):
                subprocess.run(['git','-C',directory,*args],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
            (root/'source.py').write_text('baseline\n');git('add','source.py');git('-c','core.hooksPath=/dev/null','commit','-m','fixture')
            path=root/'runtime/config/runtime_config.json';path.parent.mkdir(parents=True);path.write_text('{}')
            known={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()}
            original=subprocess.run(['bash','-c','test -z "$(git -C "$1" status --porcelain)"','preflight-fixture',directory],capture_output=True,text=True)
            self.assertEqual(original.returncode,1)
            self.assertEqual(original.stderr,'')
            audit(root,known)
            (root/'extra.py').write_text('unexpected')
            with self.assertRaises(ValueError):audit(root,known)
            (root/'extra.py').unlink();path.write_text('{"changed":true}')
            with self.assertRaises(ValueError):audit(root,known)
            path.write_text('{}');(root/'source.py').write_text('modified\n')
            with self.assertRaises(subprocess.CalledProcessError):audit(root,known)
            git('add','source.py')
            with self.assertRaises(subprocess.CalledProcessError):audit(root,known)

if __name__=='__main__':unittest.main()
