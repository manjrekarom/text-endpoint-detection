import re

def preprocess(sen):
    # Lowercase
    sen = sen.strip().lower()
    # Remove unicode chars
    sen = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", sen)
    # Remove punctuation
    sen = re.sub(r'[^\w\s]', '', sen)
    # Stemming
    # stemmer = PorterStemmer()
    # sen = " ".join([stemmer.stem(word) for word in sen.split()])

    return sen