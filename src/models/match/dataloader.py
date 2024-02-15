import h5py
from torch.utils.data import Dataset


class TranslationH5Dataset(Dataset):
    def __init__(
        self,
        path: str,
        language,
        tokenizer,
        is_val: bool = False,
        max_input_tokens: int = 125,
    ):
        """
        The training data for each group is in subgroup: language -> data
        """
        self.file_path = path
        # If the dataset is for validation, read different file
        self.is_val = is_val
        self.native_dataset_name = "val_data" if self.is_val else "data"
        # Stores the language for encoding SONAR
        self.language = language
        self.tokenizer = tokenizer
        # The files are not read here, but store boolean variable to read only once
        self.english_dataset = None
        self.spanish_dataset = None
        self.native_dataset = None
        # Max input tokens to use for tokenizer
        self.max_input_tokens = max_input_tokens
        # Get the amount of data for each language
        with h5py.File(self.file_path, "r") as file:
            self.native_length = len(file[language][self.native_dataset_name])
            self.english_length = len(file["eng"]["data"])
            self.spanish_length = len(file["spa"]["data"])

    def __len__(self) -> int:
        # The proportion of data of each language should be 1:1:1, in this
        # case english = spanish and were forcing to use #native=#english.
        # If its validation dataset, then only #native
        return (
            2 * self.english_length + self.spanish_length
            if not self.is_val
            else self.native_length
        )

    def __getitem__(self, index: int):
        # Check to which leanguage index refers to.
        if index < self.english_length:
            # If file is not yead loaded, read it
            if self.native_dataset is None:
                self.native_dataset = h5py.File(self.file_path, "r")[self.language][
                    self.native_dataset_name
                ].asstr()
            # Get the data according to the true index in native dataset.
            # Since index can be greater than the # native data we use the
            # residual to get the true index
            data = self.native_dataset[index % self.native_length]
            input = data[0]
            label = data[1]
        # Check for english
        elif index < 2 * self.english_length:
            if self.english_dataset is None:
                self.english_dataset = h5py.File(self.file_path, "r")["eng"][
                    "data"
                ].asstr()
            # Substract the length of english data to get true index
            data = self.english_dataset[index - self.english_length]
            input = data[0]
            label = data[1]
        else:  # spanish
            if self.spanish_dataset is None:
                self.spanish_dataset = h5py.File(self.file_path, "r")["spa"][
                    "data"
                ].asstr()
            data = self.spanish_dataset[index - 2 * self.english_length]
            input = data[0]
            label = data[0]

        # tokenize input
        input = self.tokenizer(
            input,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        # Apply truncation for label according to max length for SONAR ENCODER
        # max length is currently 512 tokens for SONAR.
        label = " ".join(label.split()[:100])[:512]
        # return also language to apply specific language SONAR encoder
        return input, label
