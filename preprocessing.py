import re
import unicodedata

import nltk
from bs4 import BeautifulSoup
from bs4 import Comment
from nltk import regexp_tokenize
from nltk.corpus import stopwords


# Initialize NLP parameters


# Cleaning function for new question
def cleanMe(txt):
    soup = BeautifulSoup(txt, "lxml")
    [x.extract() for x in soup.find_all('code')]
    [x.extract() for x in soup.find_all('script')]
    [x.extract() for x in soup.find_all('style')]
    [x.extract() for x in soup.find_all('meta')]
    [x.extract() for x in soup.find_all('noscript')]
    [x.extract() for x in soup.find_all(text=lambda text:isinstance(text, Comment))]
    return soup.get_text()

def remove_specific_typeOfwords(txt, nlp):
    doc = nlp(txt)
    list_text_row = []
    for token in doc:
        if(token.pos_ not in ['POS','ADJ','ADP','ADV','AUX','CONJ','CCONJ','DET','INTJ','NUM','PART','PUNCT','SCONJ','SYM','VERB','X','SPACE']):
            list_text_row.append(token.text)
    join_text_row = " ".join(list_text_row)
    join_text_row = join_text_row.lower().replace("c #", "c#")
    return join_text_row

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def text_cleaner(txt, nlp):
    # Remove specific type of word
    txt = remove_specific_typeOfwords(txt, nlp)
    # Case normalization
    txt = txt.lower()
    # Remove unicode characters
    txt = txt.encode("ascii", "ignore").decode()
    # Remove English contractions
    txt = re.sub("\'\w+", '', txt)
    # Remove accent
    txt = remove_accented_chars(txt)
    # Remove ponctuation but not # or ++
    #     txt = re.sub('[^\\w\\s#]', '', txt)
    txt = re.sub('[^\\w\\s(#|++)]', '', txt)
    # Remove links
    txt = re.sub(r'http*\S+', '', txt)
    # Remove numbers
    txt = re.sub(r'\w*\d+\w*', '', txt)
    # Remove extra spaces
    txt = re.sub('\s+', ' ', txt)
    # Tokenization with exception for C# and c++
    txt = regexp_tokenize(txt, pattern=r"\s|[\.,;']", gaps=True)
    # remove # caracter alone after tokenization
    txt = [element for element in txt if element != '#']
    #     txt = [element for element in txt if len(element) != 1]
    #     # List of stop words in select language from NLTK
    #     # Remove stop words
    stop_words = stopwords.words("english")
    #     # Remove stop words
    txt = [word for word in txt if word not in stop_words ]
    #     # Lemmatizer
    wn = nltk.WordNetLemmatizer()
    txt = [wn.lemmatize(word) for word in txt]

    return txt