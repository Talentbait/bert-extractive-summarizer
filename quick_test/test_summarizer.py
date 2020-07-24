import json
from summarizer import Summarizer
from summarizer.coreference_handler import CoreferenceHandler
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer, AutoTokenizer

class PythonPredictor:
    def __init__(self,greediness=0.4, use_coreference=False):
        handler = CoreferenceHandler(greedyness=greediness,spacy_model="de_core_news_sm")

        bertgerman_model = BertModel.from_pretrained('bert-base-german-cased', output_hidden_states=True)
        bertgerman_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

        # german_missing_tokens = ['ca.','bzw.','Du','Dein','Deinen','-','Kl.']

        # bertgerman_model = BertModel.from_pretrained('bert-base-german-cased', output_hidden_states=True,)
        # bertgerman_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased',never_split=german_missing_tokens,do_basic_tokenize=True)

        # bertgerman_model = AutoModelForTokenClassification.from_pretrained("severinsimmler/literary-german-bert")
        # bertgerman_tokenizer = AutoTokenizer.from_pretrained("severinsimmler/literary-german-bert")

        # added_tokens = bertgerman_tokenizer.add_tokens(german_missing_tokens)
        # print('We have added', added_tokens, 'tokens')
        # bertgerman_model.resize_token_embeddings(len(bertgerman_tokenizer) + len(german_missing_tokens))

        model = Summarizer(custom_model=bertgerman_model, custom_tokenizer=bertgerman_tokenizer) if not use_coreference else Summarizer(custom_model=bertgerman_model, custom_tokenizer=bertgerman_tokenizer, sentence_handler=handler)

        self.model = model

    def predict(self,body,use_first=True,max_length=500,nr_sentences=4, min_length=25,algorithm='kmeans',clusters=2,use_original=False):

        output = self.model(body, nr_sentences=nr_sentences, min_length=min_length, max_length=max_length, use_first=use_first, algorithm=algorithm, clusters=clusters, use_original=use_original)
        return output