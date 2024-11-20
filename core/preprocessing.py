import re
import pandas as pd
from typing import List
import logging
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class TextPreprocessor:
    def __init__(self):
        # Inisialisasi stemmer dan stopwords
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Load stopwords
        self.stopwords = self._load_stopwords()
        
        # Load slang dict
        self.slang_dict = self._load_slang_dictionaries()
        
        self.logger = logging.getLogger(__name__)

    def _load_stopwords(self) -> set:
        try:
            nltk_stopwords = set(stopwords.words('indonesian'))
            file_stopwords = set(pd.read_csv(
                'resources/stopwords.txt', 
                header=None
            )[0])
            return nltk_stopwords.union(file_stopwords)
        except Exception as e:
            self.logger.error(f"Error loading stopwords: {e}")
            return set()

    def _load_slang_dictionaries(self) -> dict:
        try:
            # Load the slang dictionary
            dict_1 = pd.read_csv('resources/slang-words.csv').set_index('kataAlay')
            
            dict_2 = pd.read_csv('resources/colloquial-indonesian-lexicon.csv')
            dict_2 = dict_2.filter(['slang', 'formal'], axis=1).drop_duplicates(subset=['slang'], keep='first')
            dict_2 = dict_2.set_index('slang')
            
            # Combine both dictionaries
            slang_dict_combined = pd.concat([dict_1['kataBaik'], dict_2['formal']])
            return slang_dict_combined.to_dict()
        except Exception as e:
            self.logger.error(f"Error loading slang dictionaries: {e}")
            return {}

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # Proses pembersihan teks
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Hapus URL
        text = re.sub(r'(@\w+|#\w+)', '', text)  # Hapus mention dan hashtag
        text = re.sub('<.*?>', '', text)  # Hapus HTML
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hanya huruf dan spasi

        # Normalisasi kata slang
        words = text.split()
        normalized_words = [self.slang_dict.get(word, word) for word in words]
        
        # Stemming
        stemmed_text = self.stemmer.stem(' '.join(normalized_words))
        
        # Hapus stopwords
        final_words = [
            word for word in stemmed_text.split() 
            if word not in self.stopwords and len(word) > 1
        ]

        return ' '.join(final_words)

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        return [self.clean_text(text) for text in texts]