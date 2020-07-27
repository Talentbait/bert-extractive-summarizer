from typing import List, Optional, Tuple

import numpy as np
from transformers import *

from summarizer.bert_parent import BertParent
from summarizer.cluster_features import ClusterFeatures
from summarizer.cluster_features_original import ClusterFeaturesOriginal
from summarizer.sentence_handler import SentenceHandler

import io
import os

class ModelProcessor(object):

    def __init__(
        self,
        model: str = 'bert-large-uncased',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345,
        use_original: bool = False
    ):
        """
        This is the parent Bert Summarizer model. New methods should implement this class

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param sentence_handler: The handler to process sentences. If want to use coreference, instantiate and pass CoreferenceHandler instance
        :param random_state: The random state to reproduce summarizations.
        """

        np.random.seed(random_state)
        self.model = BertParent(model, custom_model, custom_tokenizer)
        self.hidden = hidden
        self.reduce_option = reduce_option
        self.sentence_handler = sentence_handler
        self.random_state = random_state
        base_path = os.path.dirname(__file__)
        with io.open(base_path + "/teaser_sentences.txt") as input_file:
            self.base_examples = input_file.readlines()

    def process_content_sentences(self, body: str, min_length:int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences with neural coreference.
        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences with coreference applied.
        """

        doc = self.nlp(body)._.coref_resolved
        doc = self.nlp(doc)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]

    def cluster_runner(
            self,
            content: List[str],
            nr_sentences: int = 4,
            algorithm: str = 'kmeans',
            use_first: bool = True,
            clusters: int = 2,
            use_original: bool = False
    ) -> Tuple[List[str], np.ndarray]:
        """
        Runs the cluster algorithm based on the hidden state. Returns both the embeddings and sentences.

        :param content: Content list of sentences.
        :param nr_sentences: Sentences to output at summary
        :param algorithm: Type of algorithm to use for clustering.
        :param use_first: Whether to use first sentence (helpful for news stories, etc).
        :param clusters: Clusters to use from base examples
        :param use_original: Wether or not to use the original clustering approach
        :return: A tuple of summarized sentences and embeddings
        """

        hidden = self.model(content, self.hidden, self.reduce_option)

        if use_original:
            print("Using original summarazier clustering")
            hidden_args = ClusterFeaturesOriginal(hidden, algorithm, random_state=self.random_state).cluster(nr_sentences)
            
            if use_first:
                if hidden_args[0] != 0:
                    hidden_args.insert(0,0)

            sentences = [content[j] for j in hidden_args]
            embeddings = np.asarray([hidden[j] for j in hidden_args])

            return sentences, embeddings

        print("Using modified summarazier clustering")
        hidden_examples = self.model(self.base_examples, self.hidden, self.reduce_option)
        hidden_args = ClusterFeatures(hidden,hidden_examples, algorithm, random_state=self.random_state).cluster(nr_sentences + 1,clusters)

        sentences = []
        ordered_ids = []
        for j in hidden_args.values():
            ordered_ids.extend(j)
            sentences.append([content[f] for f in j])

        ordered_ids = list(dict.fromkeys(ordered_ids))
        ordered_ids.sort()

        print("ORdered IDs:\t", ordered_ids)

        if use_first and 0 not in ordered_ids:
            ordered_ids = [0] + ordered_ids
        
        if not use_first and 0 in ordered_ids:
            ordered_ids = [a for a in ordered_ids if a]

        if ordered_ids:
            sentences = [content[j] for j in ordered_ids]
        
        embeddings = np.asarray([hidden[j] for j in ordered_ids])

        return sentences, embeddings

    def __run_clusters(self, content: List[str], nr_sentences=4, algorithm='kmeans', use_first: bool= True,clusters: int = 2, use_original: bool = False) -> List[str]:
        """
        Runs clusters and returns sentences.

        :param content: The content of sentences.
        :param nr_sentences: Sentences to output at summary
        :param algorithm: Algorithm selection for clustering.
        :param use_first: Whether to use first sentence
        :param clusters: Clusters to use from base examples
        :param use_original: Wether or not to use the original clustering approach
        :return: summarized sentences
        """

        sentences, _ = self.cluster_runner(content, nr_sentences, algorithm, use_first, clusters, use_original)
        return sentences

    def __retrieve_summarized_embeddings(
            self, content: List[str], nr_sentences=4, algorithm='kmeans', use_first: bool= True, clusters: int = 2
    ) -> np.ndarray:
        """
        Retrieves embeddings of the summarized sentences.

        :param content: The content of sentences.
        :param nr_sentences: Sentences to output at summary
        :param algorithm: Algorithm selection for clustering.
        :param use_first: Whether to use first sentence
        :param clusters: Clusters to use from base examples
        :return: Summarized embeddings
        """

        _, embeddings = self.cluster_runner(content, nr_sentences, algorithm, use_first, clusters)
        return embeddings

    def run(
        self,
        body: str,
        nr_sentences: int = 4,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str ='kmeans',
        clusters: int = 2,
        use_original: bool = False
    ) -> str:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process
        :param nr_sentences: Sentences to output at summary
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param clusters: Clusters to use from base examples
        :param use_original: Wether or not to use the original clustering approach
        :return: A summary sentence
        """

        sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            sentences = self.__run_clusters(sentences, nr_sentences, algorithm, use_first, clusters, use_original)

        # return ' '.join(sentences)
        return sentences

    def run_embeddings(
        self,
        body: str,
        nr_sentences: int = 4,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str ='kmeans',
        clusters: int = 2
    ) -> Optional[np.ndarray]:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the embeddings.

        :param body: The raw string body to process
        :param nr_sentences: Sentences to output at summary
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param clusters: Clusters to use from base examples
        :return: A summary embedding
        """

        sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            embeddings = self.__retrieve_summarized_embeddings(sentences, nr_sentences, algorithm, use_first, clusters)
            return embeddings

        return None

    def __call__(
        self,
        body: str,
        nr_sentences: int = 4,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans',
        clusters: int = 2,
        use_original: bool = False
    ) -> str:
        """
        (utility that wraps around the run function)

        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process
        :param nr_sentences: Sentences to output at summary
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param clusters: Clusters to use from base examples
        :param use_original: Wether or not to use the original clustering approach
        :return: A summary sentence
        """

        return self.run(body, nr_sentences, min_length, max_length, algorithm=algorithm, use_first=use_first, clusters=clusters,use_original=use_original)



class Summarizer(ModelProcessor):

    def __init__(
        self,
        model: str = 'bert-large-uncased',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345,
        use_original: bool = False
    ):
        """
        This is the main Bert Summarizer class.

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param greedyness: associated with the neuralcoref library. Determines how greedy coref should be.
        :param language: Which language to use for training.
        :param random_state: The random state to reproduce summarizations.
        :param use_original: Wether or not to use the original clustering approach
        """

        super(Summarizer, self).__init__(
            model, custom_model, custom_tokenizer, hidden, reduce_option, sentence_handler, random_state, use_original
        )


class TransformerSummarizer(ModelProcessor):

    MODEL_DICT = {
        'Bert': (BertModel, BertTokenizer),
        'OpenAIGPT': (OpenAIGPTModel, OpenAIGPTTokenizer),
        'GPT2': (GPT2Model, GPT2Tokenizer),
        'CTRL': (CTRLModel, CTRLTokenizer),
        'TransfoXL': (TransfoXLModel, TransfoXLTokenizer),
        'XLNet': (XLNetModel, XLNetTokenizer),
        'XLM': (XLMModel, XLMTokenizer),
        'DistilBert': (DistilBertModel, DistilBertTokenizer),
    }

    def __init__(
        self,
        transformer_type: str = 'Bert',
        transformer_model_key: str = 'bert-base-uncased',
        transformer_tokenizer_key: str = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):

        try:
            self.MODEL_DICT['Roberta'] = (RobertaModel, RobertaTokenizer)
            self.MODEL_DICT['Albert'] = (AlbertModel, AlbertTokenizer)
            self.MODEL_DICT['Camembert'] = (CamembertModel, CamembertTokenizer)
        except Exception as e:
            pass # older transformer version

        model_clz, tokenizer_clz = self.MODEL_DICT[transformer_type]
        model = model_clz.from_pretrained(transformer_model_key, output_hidden_states=True)

        tokenizer = tokenizer_clz.from_pretrained(
            transformer_tokenizer_key if transformer_tokenizer_key is not None else transformer_model_key
        )

        super().__init__(
            None, model, tokenizer, hidden, reduce_option, sentence_handler, random_state
        )
