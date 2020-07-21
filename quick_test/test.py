from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
import streamlit as st
from test_summarizer import PythonPredictor
import io
import json
import spacy
import re
from summarizer.sentence_handler import SentenceHandler
from typing import List

def preprocess_jobposting(contnent: str) -> str:
    """
    Preprocesses the content of the jobposting.

    :param body: The raw string of the jobposting text
    :return: Content preprocessed.
    """

    # Add missing dot to handle large blocs with bullet point text
    contnent = re.sub(r"\n{2,}",".",contnent)
    contnent = re.sub(r"^\... ",".",contnent)

    return contnent

@st.cache()
def get_sample_texts():
    with io.open("quick_test/sample_texts.json") as json_file:
        sample_texts = json.load(json_file)
    return sample_texts

sample_texts = get_sample_texts()

def my_hash_func(a):
    return 2

st.title("BERT extract summarizer")

# @st.cache(hash_funcs={PythonPredictor: my_hash_func})
def init_query_predict(greediness, use_coreference):
    return PythonPredictor(greediness, use_coreference)

st.sidebar.subheader("Settings")

algorithm = st.sidebar.selectbox("Choose algroithm",['kmeans','gmm'],0)
use_coreference = st.sidebar.checkbox("Use Coreference handler",False)
greedy = st.sidebar.slider("Greedines",0.0,1.0,0.45,0.05)
use_first = st.sidebar.checkbox("Use first sentence",False)
min_length = st.sidebar.slider("Sentence min length",0,100,40,5)
max_length = st.sidebar.slider("Sentence max length",20,500,500,10)
ratio = st.sidebar.slider("Summarizer ratio",0.1,1.0,0.2,0.05)

selected_jobposting = st.sidebar.selectbox(
    label = "Select a sample jobposting",
    options = list(sample_texts.keys()),
    index = 4
)

predictor = init_query_predict(greedy,use_coreference)
@st.cache(hash_funcs={PythonPredictor: my_hash_func})
def get_fixed_coref_en():
    coref_enabled = PythonPredictor(greediness=0.45, use_coreference=True)
    return coref_enabled

@st.cache(hash_funcs={PythonPredictor: my_hash_func})
def get_fixed_coref_dis():
    coref_disabled = PythonPredictor(greediness=0.45, use_coreference=False)
    return coref_disabled

coref_enabled = get_fixed_coref_en()
coref_disabled = get_fixed_coref_dis()

st.subheader("Input")

jobposting = st.text_area("Text to summarize",(sample_texts[selected_jobposting]),height=500)

# jobposting = preprocess_jobposting(jobposting)

output = predictor.predict(jobposting,use_first,max_length,ratio,min_length)
st.subheader("Output")
st.write(output)

st.header("Fixed examples")

fixed_output = []
fixed_output.append(coref_enabled.predict(jobposting,use_first=True))
fixed_output.append(coref_enabled.predict(jobposting,use_first=False))
fixed_output.append(coref_disabled.predict(jobposting,use_first=True))
fixed_output.append(coref_disabled.predict(jobposting,use_first=False))

st.text_area("Coreference Enabled   First sentence Enabled", fixed_output[0],height=180)
st.text_area("Coreference Enabled   First sentence Disabled", fixed_output[1],height=180)
st.text_area("Coreference Disabled  First sentence Enabled", fixed_output[2],height=180)
st.text_area("Coreference Disabled  First sentence Disabled", fixed_output[3],height=180)



st.subheader("Tokenizer comparisson")

def filter_sentences_by_length(sentences: List[str], max_length: int = 500, min_length: int = 40):
    """
    Filter the identified sentences by the length restrictions

    :param max_lenght: maximum lenght to accept in a sentence
    :param min_lenght: minumum lenght to accept in a sentence
    :return: A tuple with the list of used and list of unused sentences
    """

    # Construct classified dict
    sentence_described = []
    for sentence in sentences:
        sentence_described.append(
            {
                'sentence':sentence,
                'used': max_length > len(sentence) > min_length
            }
        )


    used = [a['sentence'] for a in sentence_described if a['used']]
    not_used = [a['sentence'] for a in sentence_described if not a['used']]

    return used, not_used

st.subheader("Using Spacy de_core_news_sm")
nlp = spacy.load("de_core_news_sm")
spacy_sentences = [sent.text for sent in nlp(jobposting).sents]

spacy_used, spacy_not_used = filter_sentences_by_length(spacy_sentences,max_length,min_length)
st.write("Used:")
st.json(spacy_used)
st.write("Dropped:")
st.json(spacy_not_used)

st.subheader("Using SooMaJo sentence splitter")
sen_handler = SentenceHandler()
somajo_sentences = sen_handler(jobposting, min_length=0, max_length=10000)

somajo_used, somajo_not_used = filter_sentences_by_length(somajo_sentences,max_length,min_length)
st.write("Used:")
st.json(somajo_used)
st.write("Dropped:")
st.json(somajo_not_used)

st.header("Another summarizer")

from summa.summarizer import summarize
st.write(summarize(jobposting, language='german',ratio=0.20))