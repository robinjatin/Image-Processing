import spacy
from spacy.pipeline import EntityRuler
from extractacy.extract import ValueExtractor
from logger import get_logger

log = get_logger(__name__)


class Spacy:
    @staticmethod
    def perform_spacy(text):
        nlp = spacy.load("en_core_web_sm")
        # Set up entity ruler
        ruler = nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "Bill_Amount", "pattern": [{"LOWER": "previous"}, {"LOWER": "billed"}, {"LOWER": "amount"}]},
            {"label": "Last_Payment",
             "pattern": [{"LOWER": "payments"}, {"LOWER": "since"}, {"LOWER": "last"}, {"LOWER": "bill"}]},
            {"label": "Current_Electric",
             "pattern": [{"TEXT": {"REGEX": "[Current]{5,11}"}}, {"TEXT": {"REGEX": "[electric]{4,7}"}},
                         {"TEXT": {"REGEX": "[charges]{4,7}"}}]},
            {"label": "Current_Gas", "pattern": [{"LOWER": "gas"}, {"LOWER": "charges"}]},
            {"label": "Account", "pattern": [{"LOWER": "account"}, {"LOWER": "number"}, {"IS_PUNCT": True}]},
            {"label": "Total_Amount_Due",
             "pattern": [{"TEXT": {"REGEX": "[total]{4,7}"}}, {"TEXT": {"REGEX": "[amount]{4,7}"}},
                         {"TEXT": {"REGEX": "[due]{2,4}"}}, {"TEXT": {"REGEX": "[now]{2,4}"}}]},
            {"label": "Billing_Current_Date",
             "pattern": [{"LOWER": "current"}, {"LOWER": "billing"}, {"LOWER": "period"}]},
            {"label": "Late_Charges",
             "pattern": [{"LOWER": "late"}, {"LOWER": "charges"}, {"LOWER": "added"}, {"LOWER": "after"}]},
            {"label": "Energy_Used", "pattern": [{"LOWER": "electric"}, {"LOWER": "usage"}]}]
        ruler.add_patterns(patterns)

        ent_patterns = {
            "Account": {"patterns": [[{"SHAPE": "dddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"},
                                      {"ORTH": "-"}, {"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "d"}]], "n": 2,
                        "direction": "right", },
            "Last_Payment": {"patterns": [[{"ORTH": "$", "OP": "+"}, {"LIKE_NUM": True, "LENGTH": {">=": 1}}]], "n": 6,
                             "direction": "right"},
            "Bill_Amount": {"patterns": [[{"ORTH": "$", "OP": "?"}, {"LIKE_NUM": True, "LENGTH": {">=": 1}}]], "n": 1,
                            "direction": "right"},
            "Current_Gas": {"patterns": [[{"ORTH": "$", "OP": "?"}, {"LIKE_NUM": True, "LENGTH": {">=": 1}}]], "n": 1,
                            "direction": "right"},
            "Current_Electric": {"patterns": [[{"ORTH": "$", "OP": "?"}, {"LIKE_NUM": True, "LENGTH": {">=": 1}}]],
                                 "n": 1, "direction": "right"},
            "Total_Amount_Due": {"patterns": [[{"ORTH": "$", "OP": "?"}, {"LIKE_NUM": True, "LENGTH": {">=": 1}}]],
                                 "n": 1, "direction": "right"},
            "Billing_Current_Date": {"patterns": [[{"LENGTH": {">=": 1}}]], "n": 7, "direction": "right"},
            "Late_Charges": {"patterns": [[{"TEXT": {"REGEX": "[\\w]{3}"}}, {"TEXT": {"REGEX": "[\\d,]{4,7}"}}]],
                             "n": 3, "direction": "right"},
            "Energy_Used": {"patterns": [[{"TEXT": {"REGEX": "[\\d]{3,5}"}}, {"LOWER": {"IN": ["kwh", "wh"]}}]], "n": 3,
                            "direction": "right"}}
        nlp.add_pipe("valext", config={"ent_patterns": ent_patterns}, last=True)
        log.info("Extracted Key Value Pairs:")
        data = {}
        doc = nlp(text)
        for e in doc.ents:
            if e._.value_extract:
                log.info(f"{e.label_}: {e._.value_extract}")
                data[e.label_] = e._.value_extract
        return data
