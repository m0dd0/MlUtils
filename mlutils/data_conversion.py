from pathlib import Path
from typing import List

from tqdm import tqdm
import numpy as np
import h5py


class Hdf5SubsetExtractor:
    def __init__(
        self,
        hdf5_file_path: str,
        subset_hdf5_file_path: str,
        subset_indices: List[int],
        hdf5_data_group_name: str,
    ):
        self.hdf5_file_path = hdf5_file_path
        self.subset_hdf5_file_path = subset_hdf5_file_path
        self.subset_indices = subset_indices
        self.hdf5_data_group_name = hdf5_data_group_name

        self.hdf5_file = h5py.File(hdf5_file_path, "r")
        self.subset_hdf5_file = h5py.File(subset_hdf5_file_path, "w")

        self.data_group = self.hdf5_file[hdf5_data_group_name]
        self.subset_data_group = self.subset_hdf5_file.create_group(
            hdf5_data_group_name
        )

        self.subset_datasets = {
            key: self.subset_data_group.create_dataset(
                key,
                shape=(len(subset_indices),) + self.data_group[key].shape[1:],
                dtype=self.data_group[key].dtype,
            )
            for key in self.data_group.keys()
        }

    def __call__(self):
        for i, subset_idx in tqdm(enumerate(self.subset_indices)):
            for key, dataset in self.data_group.items():
                self.subset_datasets[key][i] = dataset[subset_idx]

        self.hdf5_file.close()
        self.subset_hdf5_file.close()


class NpzSubsetExtractor:
    def __init__(
        self,
        npz_folder: str,
        subset_npz_folder: str,
        subset_indices: List[int],
        datapoint_naming_format: str = "datapoint_{}.npz",
        # global_datapoint_search: bool = False,
    ):
        self.npz_folder = Path(npz_folder)
        self.subset_npz_folder = Path(subset_npz_folder)
        self.subset_indices = subset_indices
        self.datapoint_naming_format = datapoint_naming_format
        # self.global_datapoint_search = global_datapoint_search

    def __call__(self):
        # if self.global_datapoint_search:
        #     all_npz_files = list(self.npz_folder.glob("*.npz"))
        # else:
        #     all_npz_files = [
        #         self.npz_folder / self.datapoint_naming_format.format(i)
        #         for i in self.subset_indices
        #     ]

        subset_npz_files = []
        for subset_idx in self.subset_indices:
            subset_npz_file = self.npz_folder / self.datapoint_naming_format.format(
                subset_idx
            )
            if not subset_npz_file.exists():
                print(f"Datapoint {subset_idx} not found in the npz folder.")
                continue
            subset_npz_files.append(subset_npz_file)

        for subset_npz_file in tqdm(subset_npz_files):
            subset_npz_data = np.load(subset_npz_file)
            np.savez(self.subset_npz_folder / subset_npz_file.name, **subset_npz_data)


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

        if not self._npz_folder.exists():
            self._npz_folder.mkdir(parents=True)

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
                    **{
                        key: value[i_datapoint - start_idx]
                        for key, value in data_batch.items()
                    },
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
