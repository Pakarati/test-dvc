import os
import typing as tp
from collections import Counter

import fasttext

from src.mapudungun.detector_traductor_grafemario import detect_grafemario
from src.preprocess.classes import TranslationPair

VOCAB = ["eng", "spa", "arn", "rap"]
MAP = "arn"


def load_model(model_name: str) -> fasttext.FastText._FastText:
    """
    Function in charge of loading
    language detection model
    Args:
        model_name: Name to read model file from
    Returns:
        model: An instance of FastText model corresponding
                to language detection model
    """
    # load model
    print("Loading model...")
    model_path = os.path.join("models", f"{model_name}.bin")
    model = fasttext.load_model(model_path)
    return model


def get_language(predicted_langs, lang_vocab):
    """
    This functions filters out predictions from detection model,
    given if they're present or not in vocab
    Args:
        predicted_langs: List of top k predicted languages
        lang_vocab: List of languages in vocabulary
    Returns:
        predicted_lang: Returns most probable language that is in objective vocabulary.
    """
    # get language name
    predicted_langs = [t.replace("__label__", "")[:3] for t in predicted_langs]
    for lang in predicted_langs:
        # check if language is in vocab
        if lang in lang_vocab:
            return lang  # if match then return
    # if no match, return most probable
    return predicted_langs[0]


def recursive_grafemario(text, detected=[]):
    """
    Recursive function. Splits texts in half recursively until
    2 words are left. In each step detect grafemario and append
    predictions to list. This is done since detect grafemario
    function is very strict and with only one word can fail, so
    we want to get a "majority vote"
    Args:
        text: A text to predict grafemario
        detected: List of predicted grafemarios in current step
    Returns:
        detected: Predicted grafemario after each recursive step
    """
    if len(text) < 3:  # if 2 remaining words stop
        return detected
    # detect grafemarios for current text
    detected.extend(detect_grafemario(" ".join(text)))
    # split text in half and recursive call
    split = len(text) // 2
    detected = recursive_grafemario(text[:split], detected)
    detected = recursive_grafemario(text[split:], detected)
    return detected


def obtain_grafemario(text):
    """
    Main function to get mapudungun grafemario
    Args:
        text: A text to predict grafemario
    Returns:
        gram: Predicted grafemario corresponding to input
        text
    """
    # get grafemarios to each text split
    graf_detected = recursive_grafemario(text.split(" "), [])
    # count occurrences of each grafemario and get most common
    counter = Counter(graf_detected)
    prob_gram = counter.most_common(1)
    # filter that predicted grafemario is None since strict
    # model sometimes returns None
    gram, _ = prob_gram[0] if prob_gram else (None, 0)
    return gram


def predict_language(text, model):
    """
    Main language predict function
    Args:
        text: A text to predict language
        model: Language detection model to use
    Returns:
        predicted_lang: Predicted language corresponding to input
        text
    """
    detected_langs, _ = model.predict(text, k=5)  # get top k predictions
    predicted_lang = get_language(
        detected_langs, lang_vocab=VOCAB
    )  # filter predictions

    if predicted_lang == MAP:  # if mapudung get specific grafemario
        predicted_lang = obtain_grafemario(text)

    return predicted_lang


def evaluate_model(
    model: fasttext.FastText._FastText, data: tp.List[TranslationPair]
) -> tp.Tuple[tp.List, float]:
    """
    Function in charge of evaluating language detection model
    Args:
        model: A language detection model
        data: list of data entries to evaluate
    Returns:
        results: list of language detection results
        score: float corresponfing to percentage of
                correct predictions
    """
    results = []
    score = 0
    extras = []  # stores wrong language predictions
    for trans_pair in data:
        # obtain model predictiob
        predicted_lang = predict_language(trans_pair.native, model)
        # in case of correct prediction, increase score
        if predicted_lang == trans_pair.language:
            score += 1
        # otherwise save predicted language for further analysis
        else:
            extras.append(predicted_lang)

    score = (score / len(data)) * 100

    print(
        f"""
          Analized {len(data)} entries.
          Detected correctly {score}%
          """
    )

    """
    print(f"Extra langs: {Counter(extras)}")
    print(f"Extra grafemario: {Counter(extras_gram)}")
    """
    return results, score


if __name__ == "__main__":
    model = load_model("language_detector")
