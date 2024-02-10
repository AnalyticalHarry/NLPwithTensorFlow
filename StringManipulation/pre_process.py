import re
import string
from unicode_range import unicode_range

#contractions dictionary
contractions_dict = {
    "can't": "cannot",
    "can\'t": "cannot",
    "n't": " not",
    "n\'t": " not",
    "'re": " are",
    "’re": " are",
    "'s": " is",
    "’s": " is",
    "'d": " would",
    "’d": " would",
    "'ll": " will",
    "’ll": " will",
    "'t": " not",
    "’t": " not",
    "'ve": " have",
    "’ve": " have",
    "'m": " am",
    "’m": " am",
    "I'm": "I am",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "they're": "they are",
    "we're": "we are",
    "it's": "it is",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "cannot": "can not",
    "could've": "could have",
    "might've": "might have",
    "must've": "must have",
    "should've": "should have",
    "would've": "would have",
}

def preprocess_text(text):
    # function to remove emojis
    def remove_emoji(text, unicode_ranges):
        emoji_pattern = re.compile(f"[{''.join(unicode_ranges)}]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    # function to expand contractions
    def expand_contractions(text, contractions_dict):
        contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, text)

    # F=function to remove multiple consecutive spaces
    def remove_mult_spaces(text):
        return re.sub(r'\s+', ' ', text).strip()

    # function to remove symbols, hashtags, and punctuation
    def removing_symbols(text):
        text_without_hashtags = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text))
        text_without_underscores = " ".join(word.strip() for word in re.split('#|_', text_without_hashtags))
        cleaned_text = re.sub(r'[!?]', '', text_without_underscores)
        return cleaned_text

    # function to filter out words containing '$' or '&'
    def filter_char(text):
        filtered_words = [word for word in text.split() if '$' not in word and '&' not in word]
        return ' '.join(filtered_words)

    # remove emojis
    text = remove_emoji(text, unicode_range)
    # expand contractions
    text = expand_contractions(text, contractions_dict)
    # normalise whitespace and lowercase the text
    text = text.replace('\r', '').replace('\n', ' ').lower()
    # remove URLs, non-ASCII characters, and numbers
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    non_ascii_pattern = re.compile(r'[^\x00-\x7f]')
    number_pattern = re.compile(r'[0-9]+')
    text = url_pattern.sub('', text)
    text = non_ascii_pattern.sub('', text)
    text = number_pattern.sub('', text)
    # remove punctuation using translate
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove symbols, hashtags, and punctuation
    text = removing_symbols(text)
    # filter out words containing '$' or '&'
    text = filter_char(text)
    # remove multiple consecutive spaces
    text = remove_mult_spaces(text)

    return text

# single text containing various elements
test_text = """
This is a test text containing various elements such as emojis 😊, contractions like can't,
multiple spaces     and symbols! @#$ Let's see how the preprocess_text function handles all of them. #Testing123
"""

# preprocess_text function to the test text
processed_text = preprocess_text(test_text)
print("Processed text:", processed_text)

#Code Created by Hemant Thapa
#Date: 09.02.2023
#application : cleaning symbols or special characters from sentences 
#hemantthapa1998@gmail.com 
