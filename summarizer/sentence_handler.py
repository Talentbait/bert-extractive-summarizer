from typing import List
import de_core_news_sm
from spacy.attrs import ORTH, NORM
import spacy
from somajo import SoMaJo
from pprint import pprint

class SentenceHandler(object):

    def __init__(self, language=de_core_news_sm):

        german_missing_tokens = ['ca.','bzw.','Du','Dein','Deinen','-','Kl.']

        # Using Spacy module to tokeninze and split by sentences
        self.nlp = language.load()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'),before="parser")

        # Handle the tokens that are recognized as end of a sentence but are just tokens added mannually or abbreviations
        def set_custom_boundaries(doc):
            for i, token in enumerate(doc):
                if token.text in (german_missing_tokens):
                    doc[i].is_sent_start = False
            return doc

        # Implement changes from sentence boudry
        self.nlp.add_pipe(set_custom_boundaries,before="parser")

        # # Add the missing abbreviations to the vocabulary
        # for missing_token in german_missing_tokens:
        #     self.nlp.tokenizer.add_special_case(missing_token, [{ORTH:missing_token}])
        
        # Initialize the german tokeinzer
        self.tokenizer = SoMaJo("de_CMC", split_camel_case=True)


    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences must fall under
        :return: Returns a list of sentences.
        """
        
        # Using Spacy sentencizer module
        doc = self.nlp(body.replace("\r",""))
        spacy_sentences = [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]
        print("Spacy: " + str(len(spacy_sentences)))
        pprint(spacy_sentences)

        # Implementing SoMaJo tokenizer and sentencizer
        sentences = self.tokenizer.tokenize_text(body.split("\r\n"))
        
        # Unwrapping and formatting result to return sentences
        sents = []
        for sentence in sentences:
            out = []
            for token in sentence:
                if token.original_spelling is not None:
                    out.append(token.original_spelling)
                else:
                    out.append(token.text)
                if token.space_after:
                    out.append(" ")
            while out[-1] == " ":
                out = out[:-1]
            sents.append("".join(out))

        somajo_sentences = [c for c in sents if max_length > len(c) > min_length]
        print("SoMaJo: " + str(len(somajo_sentences)))
        pprint(somajo_sentences)
        print("__________________________________________________________________________________________")
        
        return somajo_sentences


    def __call__(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        return self.process(body, min_length, max_length)
