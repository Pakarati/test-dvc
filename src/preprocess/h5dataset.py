import typing as tp

import h5py
import numpy as np
from classes import TranslationPair


class h5File:
    def __init__(self, filename="data.hdf5"):
        self.filename = filename

    def _encode_data_to_numpy(self, data=tp.List[str]):
        arr = np.array(data)
        enc_arr = np.char.encode(arr)  # encode to bytes for h5py format
        return enc_arr

    def _encode_pairs_to_numpy(self, data: tp.List[TranslationPair], has_trans: bool):
        inputs = [pair.native for pair in data]
        labels = [pair.spanish for pair in data]
        data_arr = self._encode_data_to_numpy(
            list(zip(inputs, labels)) if has_trans else inputs
        )
        print(f"Result data shape: {data_arr.shape}")
        return data_arr

    def add_dataset(
        self,
        language: str,
        data: tp.List[TranslationPair],
        dataset_name: str,
        has_trans: bool = True,
    ):
        with h5py.File(self.filename, "a") as file:  # create field or read if exists
            datasets = file.keys()
            # check if language gorup is already created
            sbgrp = (
                file.create_group(language)
                if language not in datasets
                else file[language]
            )
            # check if dataset exis
            if dataset_name not in sbgrp:
                print(
                    f"Writing Dataset {dataset_name} for {language} to {self.filename}"
                )
                data_arr = self._encode_pairs_to_numpy(data, has_trans)
                # h5py asks for specic encoding stype. Enabling variable length
                utf_dtype = h5py.string_dtype(encoding="utf-8", length=None)
                sbgrp.create_dataset(
                    name=dataset_name,
                    data=data_arr,
                    shape=data_arr.shape,
                    dtype=utf_dtype,
                )
            else:
                print(
                    f"Dataset {dataset_name} for {language} \
                    already written to {self.filename}"
                )
