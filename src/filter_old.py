import re
import string
import json

REGEX = []
REGEX_EMAIL = ""
REGEX_SIMBOLS = ""
REGEX_URL = ""
DELETED_ORIGIN_PATH = ""
DELETED_TRANSLATED_PATH = ""


def number_percentage(frase):
    number_quantity = sum(1 for caracter in frase if caracter.isdigit())
    percentage = (number_quantity / len(frase)) * 100

    return percentage


def punctuation_percentage(frase):
    punctuation_set = set(string.punctuation)
    punctuation_quantity = sum(1 for caracter in frase if caracter in punctuation_set)
    percentage = (punctuation_quantity / len(frase)) * 100

    return percentage

def save_file(lines, path, format):
    f = open(path, f'{format}')

    for line in lines:
        print(line, file=f, end='')
        #agregar end='' si es necesario


def create_json(origin_lines, translated_lines, file_txt):

    list_object = []
    for origin, translated in zip(origin_lines, translated_lines):

        json_object = {"esp": translated, "arn": origin}
        list_object.append(json_object)

    with open(file_txt, 'w', encoding='utf-8') as file:
        file.write(json.dumps(list_object, ensure_ascii=True, indent=2))


def is_json(content):
    try:
        json.loads(content)
    except ValueError:
        return False
    return True

def split_text(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            json_object = json.loads(line)
            data.append(json_object)
    
    lista_esp = []
    lista_arn = []
    
    for objeto in data:
        # Extraer los valores de 'esp' y 'arn' y agregarlos a las listas respectivas
        lista_esp.append(objeto['esp'])
        lista_arn.append(objeto['arn'])
    
    return lista_arn, lista_esp

def open_txt(path):
    f = open(path)
    lines = f.readlines()
    return lines

def open_files(path):
    f = open(path)
    content = f.read()

    if content.startswith('{') and content.endswith('}'):
        return split_text(path)
    else:
        return open_txt(path)
    

def filter_lists(src, des):
    results_src = []
    results_des = []
    deleted_src = []
    deleted_des = []
    list_words = []

    for index in range(len(src)):
        new_sentence_src = ""
        new_sentence_des = ""
        m = None
        
        for reg in REGEX:
            p = re.compile(reg)
            m = p.match(src[index])
            if m != None:
                new_sentence_src = src[index][m.end():]
                new_sentence_des = des[index][m.end():]
                break
            p = re.compile(reg)
            m = p.match(des[index])
            if m != None:
                new_sentence_des = des[index][m.end():]
                break

        if m == None:
            splited = src[index].split()
            if len(splited) == 2:
                if splited[0] not in list_words:
                    list_words.append(splited[0])
                    new_sentence_src = src[index]
                    new_sentence_des = des[index]
            else:
                new_sentence_src = src[index]
                new_sentence_src = re.sub(REGEX_EMAIL, '', new_sentence_src)
                new_sentence_src = re.sub(REGEX_URL, '', new_sentence_src)
                new_sentence_src = re.sub(REGEX_SIMBOLS, '', new_sentence_src)

                new_sentence_des = des[index]
                new_sentence_des = re.sub(REGEX_EMAIL, '', new_sentence_des)
                new_sentence_des = re.sub(REGEX_URL, '', new_sentence_des)
                new_sentence_des = re.sub(REGEX_SIMBOLS, '', new_sentence_des)
        
        if len(new_sentence_src) > 1 or len(new_sentence_src) >= len(des[index])/2:
            n_per = number_percentage(new_sentence_src)
            p_per = punctuation_percentage(new_sentence_src)
            if n_per < 30 and p_per < 20:
                results_src.append(new_sentence_src)
                results_des.append(new_sentence_des)
            else:
                deleted_src.append(src[index])
                deleted_des.append(des[index])
        else:
            deleted_src.append(src[index])
            deleted_des.append(des[index])

    # Save in txt file deleted sentences
    save_file(deleted_src, DELETED_ORIGIN_PATH, 'w')
    save_file(deleted_des, DELETED_TRANSLATED_PATH, 'w')

    return results_src, results_des