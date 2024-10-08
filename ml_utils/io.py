from pathlib import Path
import datetime

import numpy as np
import cv2

from casero.array_typing import NpArray


class NpzSaver:
    def __init__(self, filename: Path, allow_overwrite: bool = True):
        self.filename = Path(filename)
        self.allow_overwrite = allow_overwrite

    def __call__(self, **data):
        if self.filename.exists() and not self.allow_overwrite:
            raise FileExistsError(f"{self.filename} already exists")

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.filename, **data)


class NpzBatchSaver:
    def __init__(
        self,
        output_dir: Path,
        file_prefix: str,
        use_datetime: bool = False,
        allow_overwrite: bool = True,
        number_of_digits: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.file_prefix = file_prefix
        self.use_datetime = use_datetime
        self.allow_overwrite = allow_overwrite
        self.number_of_digits = number_of_digits

        self.i = 0

    def __call__(self, **data) -> Path:
        if self.use_datetime:
            now = datetime.datetime.now()
            filename = now.strftime(f"{self.file_prefix}_%Y-%m-%d_%H-%M-%S.npz")
        else:
            filename = f"{self.file_prefix}{self.i:0{self.number_of_digits}}.npz"

        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        saver = NpzSaver(output_path, allow_overwrite=self.allow_overwrite)
        saver(**data)

        self.i += 1

        return output_path


class NpzLoader:
    def __init__(self, filename: Path, as_dict: bool = True):
        self.filename = Path(filename)
        self.as_dict = as_dict

    def __call__(self) -> np.lib.npyio.NpzFile:
        data = np.load(self.filename, allow_pickle=True)
        if self.as_dict:
            data = {k: data[k] for k in data.keys()}

        return data


class NpzBatchLoader:
    def __init__(
        self, input_dir: Path, ignore_subdirs: bool = True, as_dict: bool = True
    ):
        self.input_dir = Path(input_dir)
        self.ignore_subdirs = ignore_subdirs
        self.as_dict = as_dict

        if self.ignore_subdirs:
            self.npz_files = sorted(list(self.input_dir.glob("*.npz")))
        else:
            self.npz_files = sorted(list(self.input_dir.rglob("*.npz")))

        self.i = 0
        self.last_filename = None

    def __next__(self) -> np.lib.npyio.NpzFile:
        if self.i < len(self.npz_files):
            loader = NpzLoader(self.npz_files[self.i], as_dict=self.as_dict)
            self.last_filename = self.npz_files[self.i]
            self.i += 1
            return loader()
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.npz_files)


class PngSaver:
    def __init__(self, filename: Path, allow_overwrite: bool = True):
        self.filename = Path(filename)
        self.allow_overwrite = allow_overwrite

    def __call__(self, image_data: NpArray["h,w,3"], **metadata):
        if self.filename.exists() and not self.allow_overwrite:
            raise FileExistsError(f"{self.filename} already exists")

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.filename), cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
