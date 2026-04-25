import fcntl
import os
from pathlib import Path

class SingleInstanceLock:
    """Ensures only one instance of OpenCAS runs per state directory."""

    def __init__(self, state_dir: Path) -> None:
        self.lock_file = state_dir / "run.lock"
        self._fd = None

    def acquire(self) -> bool:
        """Attempt to acquire the exclusive file lock."""
        try:
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            self._fd = open(self.lock_file, "w")
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fd.write(str(os.getpid()) + "\n")
            self._fd.flush()
            return True
        except (BlockingIOError, OSError):
            if self._fd is not None:
                self._fd.close()
                self._fd = None
            return False

    def release(self) -> None:
        """Release the file lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                self._fd.close()
                self.lock_file.unlink(missing_ok=True)
            except OSError:
                pass
            self._fd = None

