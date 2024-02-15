from dataclasses import dataclass, field
from itertools import count


@dataclass
class TranslationPair:
    """
    Class for translation dataset
    Args:
        native: text in original native language
        spanish: translated text in spanish corresponding
                to native text
        language: native language used, rap or specific
                mapundung gramar
    """

    id: int = field(default_factory=count().__next__, init=False)
    native: str
    spanish: str
    language: str
