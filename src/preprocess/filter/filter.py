import json
import re
import string
import typing as tp

REGEX = [r"^\d+\.\s", r"^\d+\.\-\s", r"^-", r"^\w\)\s", r"^[-»—]+\s?"]
REGEX_EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
REGEX_URL = r"(https?://\S+|www\.\S+)"
REGEX_SIMBOLS = r"[\n»«”“]"


def filter_lists(
    origin_list: tp.List[str],
    translated_list: tp.List[str],
    save_files: bool,
    number_percentage_limit: int,
    punctuation_percentage_limit: int,
    deleted_origin_path: str,
    deleted_translated_path: str,
) -> tp.Tuple[tp.List, tp.List]:
    """
    Function that filters data lists applying regex and heuristics
    Args:
        origin_list: List of sentences in the original language
        translated_list: List of sentences in the translated language
        save_file: If the code should save txt files
        number_percentage_limit: Indicates the maximum numbers percentage allowed
        punctuation_percentage_limit: Indicates the maximum punctuation
                                      percentage allowed
    Returns:
        results_origin: Filtered data of the original language
        results_translated: Filtered data of the translated language
    """

    results_origin = []
    results_translated = []
    deleted_origin = []
    deleted_translated = []

    for index in range(len(origin_list)):
        new_sentence_origin = process_sentence(origin_list[index])
        new_sentence_translated = process_sentence(translated_list[index])

        if should_keep_sentence(
            new_sentence_origin,
            translated_list[index],
            number_percentage_limit,
            punctuation_percentage_limit,
        ):
            results_origin.append(new_sentence_origin)
            results_translated.append(new_sentence_translated)
        else:
            deleted_origin.append(origin_list[index])
            deleted_translated.append(translated_list[index])

    if save_files:
        save_file(deleted_origin, deleted_origin_path, "w")
        save_file(deleted_translated, deleted_translated_path, "w")

    return results_origin, results_translated


def process_sentence(sentence: str) -> str:
    """
    Applies regex to clean sentence
    Args:
        sentence: Sentence to clean
    Returns:
        new_sentence: Result of the sentence after applying regex
    """

    new_sentence = ""
    list_words = []

    for reg in REGEX:
        m = re.match(reg, sentence)
        if m:
            new_sentence = re.sub(reg, "", sentence)
            break

    if not new_sentence:
        splited = sentence.split()
        if len(splited) == 2 and splited[0] not in list_words:
            list_words.append(splited[0])
            new_sentence = sentence
        else:
            new_sentence = re.sub(REGEX_EMAIL, "", sentence)
            new_sentence = re.sub(REGEX_URL, "", sentence)
            new_sentence = re.sub(REGEX_SIMBOLS, "", sentence)

    return new_sentence


def should_keep_sentence(
    sentence: str,
    translated_sentence: str,
    number_percentage_limit: int,
    punctuation_percentage_limit: int,
) -> bool:
    """
    Function that applies heuristics to filter data
    Args:
        sentence: Sentence that we want to analize
        translated_sentence: Translated sentence to apply some heuristics
        number_percentage_limit: Max limit of percentage numbers in the sentence
        punctuation_precentage_limit: Max limit of percentage punctuation
                                      in the sentence
    Returns:
        Bool
    """

    if len(sentence) > 1 or len(sentence) >= len(translated_sentence) / 2:
        num_percentage = get_number_percentage(sentence)
        punctuation_percentage = get_punctuation_percentage(sentence)
        return (
            num_percentage < number_percentage_limit
            and punctuation_percentage < punctuation_percentage_limit
        )

    return False


def get_number_percentage(sentence: str) -> int:
    """
    Function that calculates the percentage of numbers in the given sentence
    Args:
        sentence: Sentence that we want to calculate the percentage
    Returns:
        percentage: Percentage result of numbers in sentence
    """
    number_quantity = sum(1 for caracter in sentence if caracter.isdigit())
    percentage = (number_quantity / len(sentence)) * 100

    return percentage


def get_punctuation_percentage(sentence: str) -> int:
    """
    Function that calculates the percentage of punctuacion in the given sentence
    Args:
        sentence: Sentence that we want to calculate the percentage
    Returns:
        percentage: Percentage result of punctuation in sentence
    """
    punctuation_set = set(string.punctuation)
    punctuation_quantity = sum(
        1 for caracter in sentence if caracter in punctuation_set
    )
    percentage = (punctuation_quantity / len(sentence)) * 100

    return percentage


def open_files(path: str, origin_lang: str, translated_lang: str) -> tp.List:
    f = open(path)
    content = f.read()

    if content.startswith("{") and content.endswith("}"):
        return split_text(path, origin_lang, translated_lang)
    else:
        return open_txt(path)


def split_text(
    path: str, origin_lang: str, translated_lang: str
) -> tp.Tuple[tp.List, tp.List]:
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            json_object = json.loads(line)
            data.append(json_object)

    origin_list = []
    translated_list = []

    for object in data:
        origin_list.append(object[origin_lang])
        translated_list.append(object[translated_lang])

    return origin_list, translated_list


def open_txt(path: str) -> tp.List:
    f = open(path)
    lines = f.readlines()

    return lines


def create_json(
    origin_lines: tp.List,
    translated_lines: tp.List,
    origin_lang: str,
    translated_lang: str,
    file_txt: str,
):
    list_object = []
    for origin, translated in zip(origin_lines, translated_lines):
        json_object = {translated_lang: translated, origin_lang: origin}
        list_object.append(json_object)

    with open(file_txt, "w", encoding="utf-8") as file:
        file.write(json.dumps(list_object, ensure_ascii=True, indent=2))


def save_file(lines: tp.List, path: str, format: str):
    f = open(path, f"{format}")

    for line in lines:
        print(line, file=f, end="")
        # add or delete end='' if needed
