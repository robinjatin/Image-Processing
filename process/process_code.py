import re
from difflib import SequenceMatcher

import nltk
import spacy
import torch
from enchant.checker import SpellChecker
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import contextualSpellCheck

from logger import get_logger

log = get_logger(__name__)
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')

NLP = spacy.load("en_core_web_sm")
nlp = spacy.load('en_core_web_md')

REP = {'\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ',
       '"': ' " ', '"': ' " ', ',': ' , ', '.': ' . ', '!': ' ! ',
       '?': ' ? ', "n't": " not", "'ll": " will", '*': ' * ',
       '(': ' ( ', ')': ' ) ', "s'": "s '", ".": ''}
REP = dict((re.escape(k), v) for k, v in REP.items())
SPELLCHECK = SpellChecker("en_US")


class TextProcessor:

    @staticmethod
    def predicted_word(text_original, predictions, mask_ids, suggested_words, tokenizer):
        # pred_words = []
        length_maskids = min(len(suggested_words), len(mask_ids))
        for i in range(length_maskids):
            preds = torch.topk(predictions[0, mask_ids[i]], k=500)
            indices = preds.indices.tolist()
            list1 = tokenizer.convert_ids_to_tokens(indices)
            list2 = suggested_words[i]
            sim_max = 0
            predicted_token = ''
            for word1 in list1:
                for word2 in list2:
                    s = SequenceMatcher(None, word1, word2).ratio()
                    if s is not None and s > sim_max:
                        sim_max = s
                        predicted_token = word1
            text_original = text_original.replace('MASK', predicted_token, 1)
            log.info(f'The word which is relavent to misspelled word is {predicted_token}')
        return text_original

    @classmethod
    def spell_check(cls, text):
        contextualSpellCheck.add_to_pipe(nlp)
        doc = nlp(text)
        print(doc._.performed_spellCheck)
        print(doc._.outcome_spellCheck)
        text = doc._.outcome_spellCheck
        log.debug(text)
        return text

    @classmethod
    def suggestion_word(cls, text):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        mask_ids = [i for i, e in enumerate(tokenized_text) if e == 'mask']
        # Create the segments tensors
        segs = [i for i, e in enumerate(tokenized_text) if e == "."]
        segments_ids = []
        prev = -1
        for k, s in enumerate(segs):
            segments_ids = segments_ids + [k] * (s - prev)
            prev = s
        segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
        segments_tensors = torch.tensor([segments_ids])
        # prepare Torch inputs
        tokens_tensor = torch.tensor([indexed_tokens])
        # Load pre-trained model
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)
        return predictions, tokenizer, mask_ids

    @staticmethod
    def text_cleaning(text):
        pattern = re.compile("|".join(REP.keys()))
        text = pattern.sub(lambda m: REP[re.escape(m.group(0))], text)
        return text

    @staticmethod
    def get_persons_list(text):
        # doc = nlp(extractedInformation)
        doc = NLP(text)
        persons_list = []
        for ent in doc.ents:
            # log.info(ent.text, ent.start_char, ent.end_char, ent.label_)
            if ent.label_ == 'PERSON':
                name = ent.text
                names = name.split()
                for name in names:
                    persons_list.append(name)
                    log.info(f'list of person name needed to be ignored as incorrect words: {persons_list}')
        return persons_list

    @classmethod
    def process(cls, file_text: str):
        text = file_text
        text = re.sub("", "", text)
        text_original = text
        # Text Cleaning
        text = cls.text_cleaning(text)

        # Ignoring Persons Name as Incorrect
        persons_list = cls.get_persons_list(text)

        ignore_words = list(persons_list) + ["!", ",", ".", "\"", "?", '(', ')', '*']
        log.info(f"ignore words: {ignore_words}")
        # using enchant.checker.SpellChecker, identify incorrect words

        words = text.split()
        incorrect_words = [w for w in words if not SPELLCHECK.check(w) and w not in ignore_words]
        # using enchant.checker.SpellChecker, get suggested replacements
        suggested_words = [SPELLCHECK.suggest(w) for w in incorrect_words]
        # Alternative method -- Contextual spell check
        # text_original = cls.spell_check(text)
        # replace incorrect words with [MASK]

        for w in incorrect_words:
            text = text.replace(w, 'MASK')
            text_original = text_original.replace(w, 'MASK')

        predictions, tokenizer, mask_ids = cls.suggestion_word(text)
        text_original = cls.predicted_word(text_original, predictions, mask_ids, suggested_words, tokenizer)
        return text_original

    @staticmethod
    def post_process(clean_text):

        final_list = []
        json_extract = {}
        doc = NLP(str(clean_text))
        count = 0
        for ent in doc.ents:
            key = ent.label_
            value = ent.text
            if key and value:
                count = count + 1
                json_extract[count] = {}
                json_extract[count][key] = value
                if len(value) > 4:
                    final_list.append(json_extract)
        log.info(f"final output below {final_list}")

    # text = text_original.lower()
    #
    # nltk.download("stopwords")
    # stop_words = set(stopwords.words("english"))
    #
    # text = " ".join([word for word in text.split() if word not in stop_words])
    #
    # # cleaning the text to remove extra whitespace
    # clean_text = " ".join([word for word in text.split()])
    # print('clean text below')
    # print(clean_text)

    # text = re.sub("@\\S+", "", clean_text)
    #
    # text = re.sub("\\$", "", text)
    #
    # text = re.sub("https?:\\\/\\\/.*[\\r\\n]*", "", text)
    # text = re.sub("#", "", text)
    # punct = set(string.punctuation)
    # text = "".join([ch for ch in text if ch not in punct])
    #
    # stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    #
    # wordList = nltk.word_tokenize(text)
    #
    # stemWords = [SnowballStemmer.stem(token=text) for word in wordList]
    # print('stem words below')
    # print(' '.join(stemWords))
    # return clean_text
# from PIL import Image
# from pytesseract import image_to_string
