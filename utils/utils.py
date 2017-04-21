import io
import sys
import contextlib

@contextlib.contextmanager
def nostdout():
  save_stdout = sys.stdout
  sys.stdout = io.BytesIO()
  yield
  sys.stdout = save_stdout
