# Import necessary libraries
import streamlit as st
import pandas as pd
import textdistance
from collections import Counter
import re

# Load Data and Preprocess
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read().lower()
            words = re.findall(r'\w+', data)
            words += words  # Add duplicates for better frequency estimation
        return words
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure the file exists and the path is correct.")
        return []

# Calculate vocabulary and probabilities
def prepare_vocab_and_probs(words):
    if not words:
        return [], {}, {}, 0
    Vocabulary = list(set(words))
    words_freq_dict = Counter(words)
    Total_word_frequency = sum(words_freq_dict.values())
    probs = {k: v / Total_word_frequency for k, v in words_freq_dict.items()}
    return Vocabulary, words_freq_dict, probs, Total_word_frequency

# Autocorrect Function
def autocorrect(input_word, Vocabulary, word_freq_dict, probs):
    input_word = input_word.lower()
    if input_word in Vocabulary:
        return None, None  # If the word is correct, skip autocorrect and suggestions
    
    similarities = [1 - textdistance.Jaccard(qval=2).distance(v, input_word) for v in word_freq_dict.keys()]
    df = pd.DataFrame({
        'Word': list(word_freq_dict.keys()),
        'Prob': list(probs.values()),
        'Similarity': similarities
    })
    output = df.sort_values(['Similarity', 'Prob'], ascending=False)  # Get the most probable autocorrect word
    return f"Autocorrect: {output.iloc[0]['Word']}", output

# Suggestion Function
def get_suggestions(input_word, Vocabulary, words_freq_dict, Total_word_frequency, top_n=5):
    input_word = input_word.lower()
    if not input_word:
        return pd.DataFrame(columns=['Word', 'Similarity', 'Frequency', 'Weighted_Score'])
    
    similarities = []
    for word in Vocabulary:
        if word:
            similarity = 1 - textdistance.Jaccard(qval=2).distance(word, input_word)
            similarities.append(similarity)
        else:
            similarities.append(0)
    
    suggestions_df = pd.DataFrame({
        'Word': Vocabulary,
        'Similarity': similarities,
        'Frequency': [words_freq_dict.get(word, 0) for word in Vocabulary]
    })
    
    suggestions_df['Weighted_Score'] = suggestions_df['Similarity'] * 0.7 + (suggestions_df['Frequency'] / Total_word_frequency) * 0.3
    suggestions_df = suggestions_df.sort_values(['Weighted_Score', 'Similarity'], ascending=False).head(top_n)
    
    return suggestions_df[['Word', 'Similarity', 'Frequency', 'Weighted_Score']]

# Load and Prepare Data
file_path = 'F:\\Jupyter\\projects\\Interactive Autocorrect and Suggestion System\\autocorrect book.txt'  # Update this path as needed
words = load_data(file_path)
Vocabulary, words_freq_dict, probs, Total_word_frequency = prepare_vocab_and_probs(words)

# Streamlit web app
st.title("Autocorrect and Suggestions")
st.write("Enter a Word to see Suggestions or Check for Autocorrect")

# User Input
input_text = st.text_input("Enter a word:", "")

if input_text:
    # Check Autocorrect Result
    autocorrect_message, autocorrect_df = autocorrect(input_text, Vocabulary, words_freq_dict, probs)
    
    # Show Autocorrect Message only if the word is incorrect
    if autocorrect_message:
        st.subheader("Autocorrect Result:")
        st.write(autocorrect_message)
    
    # Always show suggestions
    st.subheader("Suggestions:")
    suggestions_df = get_suggestions(input_text, Vocabulary, words_freq_dict, Total_word_frequency)
    st.write(suggestions_df)
