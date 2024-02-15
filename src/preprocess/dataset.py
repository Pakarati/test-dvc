import argparse
import os
import random
import typing as tp
from itertools import cycle

import nltk
from classes import TranslationPair
from h5dataset import h5File
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split


def _sample_data(data, quantity, seed):
    random.seed(seed)
    if len(data) > quantity:
        data_sample = random.sample(data, quantity)
    else:
        # shuffle data since sample is taken in order
        random.shuffle(data)
        data_sample = [next(cycle(data)) for _ in range(quantity)]

    return data_sample


def load_mono_dataset(language: str, quantity=0, seed=42) -> tp.List[TranslationPair]:
    filename = f"{language}-corpus.txt"
    print(f"Loading data from {filename}")
    with open(
        os.path.join("datasets", language, filename), "r", encoding="utf-8"
    ) as file:
        data = file.readlines()

    result_data = []
    for example in data:
        sentences = sent_tokenize(example.replace("\n", ""))
        for i in range(len(sentences)):
            entry = TranslationPair(
                native=sentences[i],
                spanish=sentences[i],
                language=language,
            )
            result_data.append(entry)

    sample = _sample_data(result_data, quantity, seed) if quantity > 0 else data

    print(
        f"""
            Loaded data for {language}
            Total texts: {len(data)}
            Total sentences: {len(result_data)}
            Final sampled data: {len(sample)}
          """
    )

    return sample


def load_pair_dataset(
    language: str, quantity=0, seed=42, spanish: str = "spa"
) -> tp.List[TranslationPair]:
    """
    Main function in charge of loading dataset and saving
    each entry as a dataclass TranslationPair
    Args:
        language: Language to read data from
    Returns:
        data: list of TranslationPair instances
    """
    native_filename = f"{language}-{spanish}-corpus.{language}"
    spanish_filename = f"{language}-{spanish}-corpus.{spanish}"
    print(f"Loading data from {native_filename}")
    with open(
        os.path.join("datasets", f"{language}-{spanish}", native_filename),
        "r",
        encoding="utf-8",
    ) as file:
        native_data = file.readlines()

    with open(
        os.path.join("datasets", f"{language}-{spanish}", spanish_filename),
        "r",
        encoding="utf-8",
    ) as file:
        spanish_data = file.readlines()

    data = []
    missed = 0
    for index in range(
        len(native_data)
    ):  # this double step is needed since data isnt in json format
        native_sent = sent_tokenize(native_data[index].replace("\n", ""))
        spanish_sent = sent_tokenize(spanish_data[index].replace("\n", ""))
        if len(native_sent) == len(spanish_sent):
            for i in range(len(native_sent)):
                entry = TranslationPair(
                    native=native_sent[i],
                    spanish=spanish_sent[i],
                    language=language,
                )
                data.append(entry)
        else:
            missed += 1

    sample = _sample_data(data, quantity, seed) if quantity > 0 else data

    print(
        f"""
            Loaded data for {spanish} - {language}
            Total texts: {len(native_data)}
            Total sentences: {len(data)}
            Missed data: {(missed/len(native_data))*100}%
            Final sampled data: {len(sample)}
          """
    )

    return sample


def build_hdf5_file(language: str, quantity: int, seed: int, results_folder: str):
    file = h5File(filename=os.path.join(results_folder, f"{language}.hdf5"))
    add_auxiliar_data(file, "spa", quantity, seed, has_trans=False)
    add_auxiliar_data(file, "eng", quantity, seed)
    add_native_data(file, args.language, seed, val_size=0.1, test_size=0.1)


def add_auxiliar_data(
    file: h5File, language: str, quantity: int, seed: int, has_trans=True
):
    # load available .txt data

    if has_trans:
        data = load_pair_dataset(language, quantity, seed)
        file.add_dataset(language, data=data, dataset_name="data")
    else:
        data = load_mono_dataset(language, quantity, seed)
        file.add_dataset(language, data=data, dataset_name="data", has_trans=False)


def add_native_data(
    file: h5File, language: str, seed: int, val_size: 0.1, test_size: 0.1
):
    native_data = load_pair_dataset(language, quantity=0, seed=seed)

    # Split data into train and test
    translation_pairs_train, translation_pairs_test = train_test_split(
        native_data,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    # split train to train and val
    translation_pairs_train, translation_pairs_val = train_test_split(
        translation_pairs_train,
        test_size=(len(native_data) * val_size) / len(translation_pairs_train),
        random_state=seed,
        shuffle=True,
    )

    print(
        f"""
        % train: {len(translation_pairs_train)/len(native_data)}
        % test: {len(translation_pairs_test)/len(native_data)}
        % validation: {len(translation_pairs_val)/len(native_data)}
          """
    )

    file.add_dataset(
        language=language, data=translation_pairs_train, dataset_name="data"
    )
    file.add_dataset(
        language=language, data=translation_pairs_val, dataset_name="val_data"
    )
    file.add_dataset(
        language=language, data=translation_pairs_test, dataset_name="test_data"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for loading and creating h5 dataset"
    )
    parser.add_argument(
        "-l",
        dest="language",
        choices=["rap", "azum", "map", "rag"],
        action="store",
        required=True,
        help="Language data to write",
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--results_folder", default="datasets", type=str)
    parser.add_argument("--quantity", default=20000, type=int)

    args = parser.parse_args()
    nltk.download("punkt")
    build_hdf5_file(
        language=args.language,
        quantity=args.quantity,
        seed=args.seed,
        results_folder=args.results_folder,
    )
