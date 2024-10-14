from pathlib import Path
import datetime

import numpy as np
import cv2
import jaxtyping as jt


class NpzSaver:
    def __init__(self, filename: Path, allow_overwrite: bool = True):
        """Save data to an npz file.

        Args:
            filename (Path): The path to the npz file.
            allow_overwrite (bool, optional): Whether to allow overwriting existing files. Defaults to True.
        """
        self.filename = Path(filename)
        self.allow_overwrite = allow_overwrite

    def __call__(self, **data):
        """Save the given data to an npz file.

        Args:
            **data: The data to save.
        """
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
        """Save data to npz files.
        Each time the saver is called, a new file is created with the given data.

        Args:
            output_dir (Path): The directory where the npz files will be saved.
            file_prefix (str): The prefix for the npz files.
            use_datetime (bool, optional): Whether to use the current datetime in the filename. Defaults to False.
            allow_overwrite (bool, optional): Whether to allow overwriting existing files. Defaults to True.
            number_of_digits (int, optional): The number of digits to use in the filename. Defaults to 4.
        """
        self.output_dir = Path(output_dir)
        self.file_prefix = file_prefix
        self.use_datetime = use_datetime
        self.allow_overwrite = allow_overwrite
        self.number_of_digits = number_of_digits

        self.i = 0

    def __call__(self, **data) -> Path:
        """Save the given data to an npz file.

        Returns:
            Path: The path to the saved npz file.
        """
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
        """Load an npz file.

        Args:
            filename (Path): The path to the npz file.
            as_dict (bool, optional): Whether to load the npz file as a dictionary
                or as a np.lib.npyio.NpzFile object. Defaults to True.
        """
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
        """An Iterator over all npz files in a directory.

        Args:
            input_dir (Path): The directory containing the npz files.
            ignore_subdirs (bool, optional): Whether to ignore subdirectories. Defaults to True.
            as_dict (bool, optional): Whether to load the npz files as dictionaries. Defaults to True.
        """
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
        """Save an image to a png file.

        Args:
            filename (Path): The path to the png file.
            allow_overwrite (bool, optional): Whether to allow overwriting existing files. Defaults to True.
        """
        self.filename = Path(filename)
        self.allow_overwrite = allow_overwrite

    def __call__(self, image_data: jt.Float[np.ndarray, "h w 3"], **metadata):
        """Save the given image to a png file.

        Args:
            image_data (jt.Float[np.ndarray, "h w 3"]): The image data.
            **metadata: Additional metadata to save with the image.

        Raises:
            FileExistsError: If the file already exists and allow_overwrite is False.
        """
        if self.filename.exists() and not self.allow_overwrite:
            raise FileExistsError(f"{self.filename} already exists")

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.filename), cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
