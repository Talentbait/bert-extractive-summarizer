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

    :param content: The raw string of the jobposting text
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

# Load the jobpostings to show as default examples for testing
@st.cache()
def get_sample_texts():
    with io.open("QUICK_TEST/sample_texts.json") as json_file:
        sample_texts = json.load(json_file)
    return sample_texts

# Model initializing functions
def init_query_predict(greediness, use_coreference):
    """
    Configure and set the Summarizer object

    :param greediness: Used for the coreference model. Anywhere from 0.35 to 0.45 seems to work well.
    :param use_coreference: Use coreference handler (not optimized) instead of the modified SetenceHandler
    :return: Instance of configured summarizer
    """
    return PythonPredictor(greediness, use_coreference)

# Model initialization using cache
@st.cache(hash_funcs={PythonPredictor: lambda _:None})
def get_fixed_coref(greediness: float = 0.45, use_coreference: bool = False):
    """
    Configure and set the Summarizer object that will not reinitialize. 
    
    :param greediness: Used for the coreference model. Anywhere from 0.35 to 0.45 seems to work well.
    :param use_coreference: Use coreference handler (not optimized) instead of the modified SetenceHandler
    :return: Instance of configured summarizer stored in cache
    """
    coref_enabled = PythonPredictor(greediness=greediness, use_coreference=use_coreference)
    return coref_enabled



st.title("BERT extract summarizer")



##########################  Streamlit desplay top  ##########################

# Load default examples from file
sample_texts = get_sample_texts()
st.sidebar.subheader("Select an example to test")

selected_jobposting = st.sidebar.selectbox(
    label = "You can edit or enter your jobposting content but when choosing a different example, it will be displayed",
    options = list(sample_texts.keys()),
    index = 4
)

# Reading input from user or prefilled example selected from the default examples
st.subheader("Input")
jobposting = st.text_area("Text to summarize",preprocess_jobposting(sample_texts[selected_jobposting]),height=500)
jobposting = preprocess_jobposting(jobposting)

# Initializing and counting the sentences of the current input
sen_handler = SentenceHandler()
somajo_sentences = sen_handler(jobposting, min_length=0, max_length=10000)


############################  Streamlit sidebar  ##############################

st.sidebar.subheader("Settings")

algorithm = st.sidebar.selectbox("Choose algroithm",['kmeans','gmm'],0)
use_coreference = st.sidebar.checkbox("Use Coreference handler",False)
use_first = st.sidebar.checkbox("Use first sentence",True)
use_original = st.sidebar.checkbox("Original summarizer algorithm", False)
greedy = st.sidebar.slider("Greediness",0.0,1.0,0.45,0.05)
min_length = st.sidebar.slider("Sentence min length",0,100,30,5)
max_length = st.sidebar.slider("Sentence max length",20,500,500,10)
ratio = st.sidebar.slider("Select a ratio (optional)",0.05,1.0,0.15,0.05)

# Calulate the sentences wanted from the ratio selected
sentences_from_ratio = int(max(ratio * len(somajo_sentences), 1))

nr_sentences = st.sidebar.slider("Output sentences (actual input)",1,10,sentences_from_ratio,1)
clusters = st.sidebar.slider("Clusters to use",1,6,2,1)


##########################  Streamlit output desplay  ##########################

st.subheader("Output")
predictor = init_query_predict(greedy,use_coreference)
output = predictor.predict(body=jobposting, use_first=use_first, max_length=max_length, nr_sentences=nr_sentences, min_length=min_length, algorithm=algorithm, clusters=clusters, use_original=use_original)
st.write(output)

st.write(" ".join(output))

##########################  Possible configruations  ###########################
st.header("Possible configruations")
if st.checkbox("Show the output for dofferent configurations"):
    fixed_output = []

    # Loading the models (just the first time)
    coref_enabled = get_fixed_coref(greediness=greedy,use_coreference=True)
    coref_disabled = get_fixed_coref(greediness=greedy,use_coreference=False)

    # Display output
    fixed_output.append(" ".join(coref_enabled.predict(jobposting,use_first=True,max_length=max_length, nr_sentences=nr_sentences, min_length=min_length, algorithm=algorithm, clusters=clusters, use_original=use_original)))
    st.text_area("Coreference ✅  First sentence ✅", fixed_output[0],height=180)

    fixed_output.append(" ".join(coref_enabled.predict(jobposting,use_first=False,max_length=max_length, nr_sentences=nr_sentences, min_length=min_length, algorithm=algorithm, clusters=clusters, use_original=use_original)))
    st.text_area("Coreference ✅  First sentence 🚫", fixed_output[1],height=180)

    fixed_output.append(" ".join(coref_disabled.predict(jobposting,use_first=True,max_length=max_length, nr_sentences=nr_sentences, min_length=min_length, algorithm=algorithm, clusters=clusters, use_original=use_original)))
    st.text_area("Coreference 🚫  First sentence ✅", fixed_output[2],height=180)

    fixed_output.append(" ".join(coref_disabled.predict(jobposting,use_first=False,max_length=max_length, nr_sentences=nr_sentences, min_length=min_length, algorithm=algorithm, clusters=clusters, use_original=use_original)))
    st.text_area("Coreference 🚫  First sentence 🚫", fixed_output[3],height=180)


###########################  Tokenizer comparisson  ############################
st.subheader("Tokenizer comparisson")

def filter_sentences_by_length(sentences: List[str], max_length: int = 500, min_length: int = 30, sentence_handler: str = "Sentence segmentator"):
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


    if st.checkbox("Show used for " + sentence_handler):
        st.json(used)

    if st.checkbox("Show not used for " + sentence_handler):
        st.json(not_used)

    return used, not_used

st.subheader("Using SooMaJo sentence splitter")
filter_sentences_by_length(somajo_sentences,max_length,min_length, "SoMaJo")

st.subheader("Using Spacy de_core_news_sm")
nlp = spacy.load("de_core_news_sm")
spacy_sentences = [sent.text for sent in nlp(jobposting).sents]
spacy_used, spacy_not_used = filter_sentences_by_length(spacy_sentences,max_length,min_length, "Sapcy")


#############################   Other approaches   #############################
st.header("Another summarizer")

from summa.summarizer import summarize
st.write(summarize(jobposting, language='german',ratio=0.20))