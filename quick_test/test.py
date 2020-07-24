from summa.summarizer import summarize
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
    contnent = re.sub(r"\s+\*\s+","\n",contnent)
    contnent = re.sub(r"\s\n\s+","\n",contnent)
    contnent = re.sub(r"als(\.\.\.)*\s+","als ",contnent)
    contnent = re.sub(r"\.*\s*\n+",".\n\n",contnent)
    for punctuation in [r"\.",r":",r"\?",r"!"]:
        contnent = re.sub(punctuation + r"\.",punctuation,contnent)

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
use_first = st.sidebar.checkbox("Use first sentence",True)
min_length = st.sidebar.slider("Sentence min length",0,100,40,5)
max_length = st.sidebar.slider("Sentence max length",20,500,500,10)
nr_sentences = st.sidebar.slider("Output sentences",1,16,4,1)
clusters = st.sidebar.slider("Clusters to use",1,6,2,1)

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


st.subheader("Input")

# st.text_area("Text to summarize",(sample_texts[selected_jobposting]),height=500)
jobposting = st.text_area("Text to summarize",preprocess_jobposting(sample_texts[selected_jobposting]),height=500)

# jobposting = preprocess_jobposting(jobposting)

output = predictor.predict(body=jobposting, use_first=use_first, max_length=max_length, nr_sentences=nr_sentences, min_length=min_length, algorithm=algorithm, clusters=clusters)
st.subheader("Output")
st.write(output)

st.write(" ".join(output))

# st.header("Fixed examples")

# fixed_output = []

# coref_enabled = get_fixed_coref_en()
# coref_disabled = get_fixed_coref_dis()

# fixed_output.append(" ".join(coref_enabled.predict(jobposting,use_first=True)))
# fixed_output.append(" ".join(coref_enabled.predict(jobposting,use_first=False)))
# fixed_output.append(" ".join(coref_disabled.predict(jobposting,use_first=True)))
# fixed_output.append(" ".join(coref_disabled.predict(jobposting,use_first=False)))

# st.text_area("Coreference Enabled   First sentence Enabled", fixed_output[0],height=180)
# st.text_area("Coreference Enabled   First sentence Disabled", fixed_output[1],height=180)
# st.text_area("Coreference Disabled  First sentence Enabled", fixed_output[2],height=180)
# st.text_area("Coreference Disabled  First sentence Disabled", fixed_output[3],height=180)



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

    st.write("Used:")
    st.json(used)
    st.write("Dropped:")
    st.json(not_used)

    return used, not_used


st.subheader("Using SooMaJo sentence splitter")
sen_handler = SentenceHandler()
somajo_sentences = sen_handler(jobposting, min_length=0, max_length=10000)
filter_sentences_by_length(somajo_sentences,max_length,min_length)

st.subheader("Using Spacy de_core_news_sm")
nlp = spacy.load("de_core_news_sm")
spacy_sentences = [sent.text for sent in nlp(jobposting).sents]
spacy_used, spacy_not_used = filter_sentences_by_length(spacy_sentences,max_length,min_length)

st.header("Another summarizer")

st.write(summarize(jobposting, language='german',ratio=0.20))