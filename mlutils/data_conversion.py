from pathlib import Path

from tqdm import tqdm
import numpy as np
import h5py


class Hdf5ToNpzDatasetConverter:
    def __init__(
        self,
        hdf5_file_path: str,
        npz_folder: str,
        hdf5_data_group_name: str,
        batch_size: int = 100,
        file_prefix: str = "datapoint",
    ):
        self._hdf5_file_path = hdf5_file_path
        self._npz_folder = Path(npz_folder)
        self._batch_size = batch_size
        self._file_prefix = file_prefix

        self._hdf5_file = h5py.File(hdf5_file_path, "r")
        self._data_group = self._hdf5_file[hdf5_data_group_name]

        # get all datasets in the group
        self._datasets = {key: self._data_group[key] for key in self._data_group.keys()}
        datset_lens = [len(dataset) for dataset in self._datasets.values()]
        assert len(set(datset_lens)) == 1, "All datasets must have the same length"
        self._n_data_points = datset_lens[0]
        self._n_batches = self._n_data_points // batch_size + 1

    def __call__(self):
        for i_batch in tqdm(range(self._n_batches)):
            start_idx = i_batch * self._batch_size
            end_idx = min((i_batch + 1) * self._batch_size, self._n_data_points)

            data_batch = {
                key: dataset[start_idx:end_idx]
                for key, dataset in self._datasets.items()
            }

            for i_datapoint in range(start_idx, end_idx):
                npz_file_path = (
                    self._npz_folder / f"{self._file_prefix}_{i_datapoint}.npz"
                )
                np.savez(
                    npz_file_path,
                    **{key: value[i_datapoint] for key, value in data_batch.items()},
                )


class NpzToHdf5DatasetConverter:
    def __init__(
        self, npz_folder: str, hdf5_file_path: str, hdf5_group_name: str = "data"
    ):
        self.npz_folder = Path(npz_folder)
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_group_name = hdf5_group_name
        self.hdf5_file = h5py.File(hdf5_file_path, "w")

        data_group = self.hdf5_file.create_group(self.hdf5_group_name)
        pivot_npz_data_path = list(self.npz_folder.glob("*.npz"))[0]

        self.hdf5_datasets = {}
        pivot_npz_data = np.load(pivot_npz_data_path)

        for key, value in pivot_npz_data.items():
            dataset = data_group.create_dataset(
                key,
                shape=value.shape,
                dtype=value.dtype,
                maxshape=(None,) + value.shape,
            )
            self.hdf5_datasets[key] = dataset

    def __call__(self):
        for i, npz_file_path in tqdm(enumerate(list(self.npz_folder.glob("*.npz")))):
            npz_data = np.load(npz_file_path)

            for key, value in npz_data.items():
                if key not in self.hdf5_datasets:
                    print(
                        f"Key {key} not found in pivot npz data. Please ensure that all npz files have the same keys."
                    )
                    continue
                hdf5_dataset = self.hdf5_datasets[key]
                hdf5_dataset.resize(hdf5_dataset.shape[0] + 1, axis=0)
                hdf5_dataset[-1, :] = value

        self.hdf5_file.close()


class NpzToPetastormConverter:
    def __init__(self, npz_folder, petastorm_folder):
        self.npz_folder = npz_folder
        self.petastorm_folder = petastorm_folder

    def __call__(self):
        raise NotImplementedError
