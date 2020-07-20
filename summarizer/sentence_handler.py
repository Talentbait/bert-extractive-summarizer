from typing import List
from spacy.lang.en import English
import de_core_news_sm
from spacy.attrs import ORTH, NORM
import spacy
from somajo import SoMaJo


class SentenceHandler(object):

    def __init__(self, language=de_core_news_sm):
        # self.nlp = language.load()
        # self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'),before="parser")
   
        german_missing_tokens = ['ca.','bzw.','Du','Dein','Deinen','-','Kl.']

        # def set_custom_boundaries(doc):
        #     for i, token in enumerate(doc):
        #         if token.text in (german_missing_tokens):
        #             doc[i].is_sent_start = False
        #     return doc
   
        # self.nlp.add_pipe(set_custom_boundaries,before="parser")

        # for missing_token in german_missing_tokens:
        #     self.nlp.tokenizer.add_special_case(missing_token, [{ORTH:missing_token}])
        
        self.tokenizer = SoMaJo("de_CMC", split_camel_case=True)


    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        # doc = self.nlp(body)
        # return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]
        sentences = self.tokenizer.tokenize_text(body.splitlines())
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
            if out[-1] == " ":
                out = out[:-1]
            sents.append("".join(out))

        return [c for c in sents if max_length > len(c) > min_length]


    def __call__(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        return self.process(body, min_length, max_length)
