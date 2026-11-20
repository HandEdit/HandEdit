"""Frame naming and output handling shared by the preparation scripts."""

import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def index_frames(directory):
    frames = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in frames:
            raise ValueError(f"Duplicate frame stem in {directory}: {path.stem}")
        frames[path.stem] = path
    if not frames:
        raise ValueError(f"No JPG or PNG frames in {directory}")
    numbers = {stem: re.search(r"\d+", stem) for stem in frames}
    if all(numbers.values()):
        return dict(sorted(frames.items(), key=lambda item: int(numbers[item[0]].group())))
    return frames


def require_same_frames(reference, other, label):
    missing = sorted(reference.keys() - other.keys())
    extra = sorted(other.keys() - reference.keys())
    if missing or extra:
        raise ValueError(
            f"{label}: missing {len(missing)} frames {missing[:5]}; "
            f"unexpected {len(extra)} frames {extra[:5]}"
        )


@contextmanager
def new_output(directory):
    directory = directory.absolute()
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Output already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".handedit-", dir=directory.parent) as temp:
        staging = Path(temp) / "output"
        staging.mkdir()
        yield staging
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Output already exists: {directory}")
        staging.rename(directory)
