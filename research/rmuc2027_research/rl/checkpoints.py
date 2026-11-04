"""Crash-safe checkpoint replacement with a previous complete generation."""
import os
from pathlib import Path
import shutil
import tempfile


def save_checkpoint(model, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.checkpoint-', dir=destination.parent) as work:
        pending = Path(work)/'model.zip'
        model.save(pending)
        with pending.open('rb') as stream:
            os.fsync(stream.fileno())
        if destination.exists():
            previous = Path(work)/'previous.zip'
            shutil.copyfile(destination, previous)
            with previous.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(previous, destination.with_suffix('.previous.zip'))
        os.replace(pending, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
