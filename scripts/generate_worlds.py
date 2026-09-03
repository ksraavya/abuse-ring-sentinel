from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for world in ("world_a", "world_b"):
        subprocess.run(
            [sys.executable, "-m", "worlds.generator", "--world", world],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
