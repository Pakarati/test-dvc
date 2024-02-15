# coding: utf-8
import re

from src.mapudungun.KimamWirintukun import chuchiWirintukun, zulliafiel_rulpawirintukuwe


def grafemario_code_to_id(grafemario_code):
    if grafemario_code == "u0":  # Ragileo
        return 1
    if grafemario_code == "r0":  # Unificado
        return 2
    if grafemario_code == "a0":  # Azümchefe
        return 3
    return None


def translate_grafemario(txt, src_id, dst_id):
    # 'a0'  : 'Azümchefe'
    # 'r0'  : 'Ragileo'
    # 'u0'  : 'Unificado'
    # 'av1' : 'Azümchefe + Tx -> Tr' --> Only supported as source text
    # 'av2' : 'Azümchefe + G  -> Ng' --> Only supported as source text
    # 'rv1' : 'Ragileo + C -> Ch' --> Only supported as source text
    # 'rv2' : 'Ragileo + V -> Ü' --> Only supported as source text
    # 'rv3' : 'Ragileo + Z -> D' --> Only supported as source text
    # 'uv1' : 'Unificado + D -> Z' --> Only supported as source text
    # 'uv2' : 'Unificado + Tr -> Tx' --> Only supported as source text

    if src_id == "rv1":
        txt = (
            txt.replace("ch", "c")
            .replace("Ch", "C")
            .replace("CH", "C")
            .replace("cH", "c")
        )
        src_id = "r0"
    if src_id == "rv2":
        txt = txt.replace("ü", "v").replace("Ü", "V")
        src_id = "r0"
    if src_id == "rv3":
        txt = txt.replace("d", "z").replace("D", "Z")
        src_id = "r0"
    if src_id == "uv1":
        txt = txt.replace("z", "d").replace("Z", "D")
        src_id = "u0"
    if src_id == "uv2":
        txt = (
            txt.replace("tx", "tr")
            .replace("Tx", "Tr")
            .replace("TX", "TR")
            .replace("tX", "tR")
        )
        src_id = "u0"
    if src_id == "av1":
        txt = (
            txt.replace("tr", "tx")
            .replace("Tr", "Tx")
            .replace("TR", "TX")
            .replace("tR", "tX")
        )
        src_id = "a0"
    if src_id == "av2":
        txt = re.sub(
            "(?<![Nn])g", "q", txt
        )  # replace all "g" not preceded by an "n" and replace them by a "q"
        txt = re.sub(
            "(?<![Nn])G", "Q", txt
        )  # replace all "g" not preceded by an "n" and replace them by a "q"
        txt = (
            txt.replace("ng", "g")
            .replace("Ng", "G")
            .replace("NG", "G")
            .replace("nG", "g")
        )
        # txt = re.sub('g','',txt) # I'm not sure if this should be commented or not.
        # I guess it should
        src_id = "a0"
    if dst_id in ["a0", "r0", "u0"] and src_id in ["a0", "r0", "u0"]:
        src_idx = grafemario_code_to_id(src_id)
        dst_idx = grafemario_code_to_id(dst_id)
        return zulliafiel_rulpawirintukuwe(src_idx, dst_idx)(txt)
    return None


COD2GRAM = {
    "a0": "azum",
    "r0": "rag",
    "u0": "map",
    "av1": "azum",
    "av2": "azum",
    "rv1": "rag",
    "rv2": "rag",
    "rv3": "rag",
    "uv1": "map",
    "uv2": "map",
}


def detect_grafemario(txt):
    # 'a0'  : 'Azümchefe'
    # 'r0'  : 'Ragileo'
    # 'u0'  : 'Unificado'
    # 'av1' : 'Azümchefe + Tx -> Tr'
    # 'av2' : 'Azümchefe + G  -> Ng'
    # 'rv1' : 'Ragileo + C -> Ch'
    # 'rv2' : 'Ragileo + V -> Ü'
    # 'rv3' : 'Ragileo + Z -> D'
    # 'uv1' : 'Unificado + D -> Z'
    # 'uv2' : 'Unificado + Tr -> Tx'

    # Returns an array of all matching grafemarios' ids

    txt = txt.strip().lower()
    # elimina todo lo que no sea letras o underscore
    txt = re.sub(r"[0-9\W]", "", txt)  # noseq W605
    if len(txt) == 0:
        return []
    tvfa = chuchiWirintukun(txt)

    """
    grams = [COD2GRAM[s[0]] for s in tvfa]
    prob_gram = Counter(grams).most_common(1)
    gram, _ = prob_gram[0] if prob_gram else (None, 0)
    return gram
    """

    return [COD2GRAM[s[0]] for s in tvfa]


"""
text = "Tüfachi pichilifru, kiñe werkün rume feleferpual, pepi trawaiñ montual
ñi poyewün, trawaiñ üllkantual, rakiduamal ka pewmatual Fachantü ka rume felerpual"

print(detect_grafemario(text))


print(translate_grafemario(text, 'u0', 'a0'))

"""
