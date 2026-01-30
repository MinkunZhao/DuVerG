import subprocess
import tempfile
import os
import sys
import signal
from contextlib import contextmanager


class CodeSandbox:
    def __init__(self, use_docker=False, timeout=60):
        self.use_docker = use_docker
        self.timeout = timeout

    def execute(self, code: str):
        if "rm -rf" in code or "shutil.rmtree" in code:
            return False, "Security Error: File deletion operations are not allowed."

        return self._execute_local(code)

    def _execute_local(self, code: str):
        fd, script_path = tempfile.mkstemp(suffix='.py', text=True)

        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(code)

            env = os.environ.copy()

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
            )

            output = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return True, output.strip()
            else:
                return False, f"Runtime Error:\n{result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            return False, f"Execution Timed Out (> {self.timeout}s)."
        except Exception as e:
            return False, f"System Error: {str(e)}"
        finally:
            if os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except:
                    pass