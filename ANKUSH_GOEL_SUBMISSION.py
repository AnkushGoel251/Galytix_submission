from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import gdown
import os

# Define the Google Drive URL for the Word2Vec vectors
google_drive_url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
download_file_path = "word2vec.bin"

# Download the file from Google Drive if it doesn't exist
if not os.path.exists(download_file_path):
    gdown.download(google_drive_url, download_file_path, quiet=False)


limit = 1000000  # Limit to the first million vectors
location = "word2vec.bin"

wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=limit)
wv.save_word2vec_format('vectors.csv', binary=False)


phrases_df = pd.read_csv('phrases.csv')

# Function to calculate the cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Process phrases and calculate similarities
similarities = {}
for idx, row in phrases_df.iterrows():
    phrase = row['Phrase']
    phrase_vector = np.zeros(wv.vector_size)
    num_words = 0

    # Calculate the vector for the phrase as the sum of word vectors
    for word in phrase.split():
        if word in wv:
            phrase_vector += wv[word]
            num_words += 1

    if num_words > 0:
        phrase_vector /= num_words

    # Calculate cosine similarity with all other phrases
    similarities[phrase] = {}
    for idx2, row2 in phrases_df.iterrows():
        other_phrase = row2['Phrase']
        if other_phrase != phrase:
            other_vector = np.zeros(wv.vector_size)
            num_other_words = 0
            for word in other_phrase.split():
                if word in wv:
                    other_vector += wv[word]
                    num_other_words += 1
            if num_other_words > 0:
                other_vector /= num_other_words
                similarity = cosine_similarity(phrase_vector, other_vector)
                similarities[phrase][other_phrase] = similarity



# Function to find the closest match to a user-input phrase
def find_closest_match(user_input):
    input_vector = np.zeros(wv.vector_size)
    num_words = 0
    for word in user_input.split():
        if word in wv:
            input_vector += wv[word]
            num_words += 1
    if num_words > 0:
        input_vector /= num_words

    best_match = None
    best_similarity = -1.0

    for phrase, similarity_dict in similarities.items():
        for other_phrase, similarity in similarity_dict.items():
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = phrase

    return best_match, best_similarity

# Test function

user_input = "ankush work to make it closest to what was asked"
closest_match, similarity = find_closest_match(user_input)
print(f"Closest match: {closest_match}, Similarity: {similarity}")
