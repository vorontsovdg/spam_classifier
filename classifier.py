import csv
import nltk
import string
import re
nltk.download('popular', './nltk_data')
nltk.data.path.append('./nltk_data')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from math import log

ALL = []
SPAM = {}
NOT_SPAM = {}
WORDBASE = {}

def send_info(func):
    def wrapper():
        print('Model training started')
        result = func()
        print('Model has finished learning')
        return result
    return wrapper



def preprocess(text):
    #lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    sw = stopwords.words('english')
    text = text.lower().translate(str.maketrans(dict.fromkeys(string.punctuation)))
    text = [word.strip() for word in text.split()]
    text = [re.sub('\d+', 'number', word) for word in text] 
    #return [lemmatizer.lemmatize(word) for word in text if not word in sw]
    return [stemmer.stem(word) for word in text if not word in sw]


@send_info
def train():
    with open('./spam_or_not_spam.csv', 'r') as f:
        csvReader = csv.DictReader(f, delimiter = ',')
        for row in csvReader:
            ALL.append(int(row['label']))
            if row['label'] == '0':
                for word in preprocess(row['email']):
                    WORDBASE.setdefault(word, 0)
                    NOT_SPAM[word] = NOT_SPAM.get(word, 0) + 1
            elif row['label'] == '1':
                for word in preprocess(row['email']):
                    WORDBASE.setdefault(word, 0)
                    SPAM[word] = SPAM.get(word, 0) + 1

def calculate_P_Bi_A(word, label):
    if label == 1:
        return log((SPAM.get(word, 0) + 1) / (sum(SPAM.values()) + len(WORDBASE)))
    elif label == 0:
        return log((NOT_SPAM.get(word, 0) + 1) / (sum(NOT_SPAM.values()) + len(WORDBASE)))

def calculate_P_B_A(text, label):
    if label == 1:
        probability = log(sum(ALL) / len(ALL))
        probability += sum([calculate_P_Bi_A(word, label) for word in preprocess(text)])
        return probability
    elif label == 0:
        probability = log((len(ALL) - sum(ALL)) / len(ALL))
        probability += sum([calculate_P_Bi_A(word, label) for word in preprocess(text)])
        return probability

def classify(email):
    pSpam = calculate_P_B_A(email, 1)
    pNotSpam = calculate_P_B_A(email, 0)
    return pSpam > pNotSpam

