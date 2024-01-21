import math
import os
import spacy
import pickle
from datasets import load_dataset
from typing import Dict, List

# Constants
ENGLISH_MODEL = "en_core_web_sm"
PICKLE_FILE_PATH = "train_lemmas.pkl"
QUESTION2_A = "I have a house in"
QUESTION3_A = "Brad Pitt was born in Oklahoma"
QUESTION3_B = "The actor was born in USA"
LAMBDA_BIGRAM = 2 / 3
LAMBDA_UNIGRAM = 1 / 3
START = "sta!rt"


def lemmatize_text(texts: List[str]) -> List[str]:
    lemmatize_list = []
    nlp = spacy.load(ENGLISH_MODEL)
    for sentence in texts:
        doc = nlp(sentence)
        sentence_lemmas = [token.lemma_ for token in doc if token.is_alpha]
        lemmatize_list.append(sentence_lemmas)
    return lemmatize_list[0]


def preprocess_lemmas(output_file_path: str = PICKLE_FILE_PATH) -> None:
    """
    Preprocesses the training dataset and saves the lemmatized tokens to a file.

    Args:
        output_file_path: (str): Path to save the processed data.
    """
    if not os.path.exists(output_file_path):
        nlp = spacy.load(ENGLISH_MODEL)
        train_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
        processed_train_lemmas = []
        for text_entry in train_data:
            if text_entry['text']:
                doc = nlp(text_entry['text'])
                lemmas = [token.lemma_ for token in doc if token.is_alpha]
                processed_train_lemmas.append(lemmas)
        with open(output_file_path, 'wb') as f:
            pickle.dump(processed_train_lemmas, f)


def train_unigram_model(log_it: bool) -> Dict[str, float]:
    """
    Trains the unigram model - makes a dictionary with the probability of word being in a text
    :return: A dictionary representing the unigram model -
             the presents of finding a specific word in the corpus
    """
    words_counts = 0
    words_appearances = {}
    with open('train_lemmas.pkl', 'rb') as f:
        corpus = pickle.load(f)

    for sentence in corpus:
        for word in sentence:
            words_counts += 1
            words_appearances[word] = words_appearances.get(word, 0) + 1
    if log_it:
        return {word: math.log(words_appearances[word] / words_counts for word in words_appearances.keys())}
    return {word: words_appearances[word] / words_counts for word in words_appearances.keys()}


def train_bigram_model(log_it: bool, model_unigram: Dict[str, float], file_path: str = "train_lemmas.pkl") -> Dict[
    str, Dict[str, float]]:
    """
    Train a bigram model based on lemmatized tokens from a given pickle file and a unigram model.

    Args:
        model_unigram (Dict[str, float]): Unigram model with word probabilities.
        file_path (str): Path to the pickle file containing lemmatized tokens.

    Returns:
        Dict[str, Dict[str, float]]: Bigram model with conditional probabilities.
    """
    # Load the lemmatized tokens from the pickle file
    with open(file_path, 'rb') as f:
        lemmatized_tokens = pickle.load(f)

    bigram_model = {}
    words_dic = {}

    # Iterate over each sentence in the lemmatized tokens
    for sentence in lemmatized_tokens:
        word1 = START
        # Iterate over each word in the sentence
        for i in range(len(sentence)):
            if word1 not in words_dic:
                words_dic[word1] = 1
            else:
                words_dic[word1] += 1
            word2 = sentence[i]

            # If word1 is not in the bigram model, initialize it
            if word1 not in bigram_model:
                bigram_model[word1] = {}

            # If word2 is not in the bigram model for word1, initialize it
            if word2 not in bigram_model[word1]:
                bigram_model[word1][word2] = 0

            # Increment the count for the word pair
            bigram_model[word1][word2] += 1
            word1 = word2

    for word1 in bigram_model.keys():
        for word2 in bigram_model[word1].keys():
            prob = bigram_model[word1][word2] / words_dic[word1]
            bigram_model[word1][word2] = math.log(prob) if log_it else prob
    return bigram_model


def bigram_model_predict_next_word(bigram_model: dict[str, dict[str, float]], sentence: List[str]) -> str:
    """

    :param bigram_model: The trained bigram model
    :param sentence: A sentence which we want to predict it's next word
    :return: The word with the highest probability to be the next word
    """
    highest_probability = 0
    fittest_word = ""
    next_word_options = bigram_model[sentence[-1]]  # The dict of next word options relevant for the sentence last word
    for n_word in next_word_options.keys():
        if next_word_options[n_word] > highest_probability:
            highest_probability = next_word_options[n_word]
            fittest_word = n_word
    return fittest_word


def bigram_model_compute_sentence_probability(bigram_model: dict[str, dict[str, float]], sentence: List[str],
                                              log_it=False) -> float:
    probability = 1.0
    word1 = START
    for i in range(len(sentence)):
        word2 = sentence[i]
        if word1 not in bigram_model or word2 not in bigram_model[word1]:
            return float('-inf')
        probability *= bigram_model[word1][word2]
        word1 = word2

    return math.log(probability) if log_it else probability


def bigram_model_compute_sentence_perplexity(bigram_model, m_sentences: List[List[str]]) -> float:
    m = len(m_sentences)  # The number of sentences in the test set
    M = 0  # Amount of words in the test data
    sum_of_sentences_log_probabilities = 0

    for sentence in m_sentences:
        M += len(sentence)  # Add the length of the current sentence
        sentence_log_probability = bigram_model_compute_sentence_probability(bigram_model, sentence, True)

        # Check for negative infinity (log probability of 0)
        if sentence_log_probability == float('-inf'):
            return float('inf')

        sum_of_sentences_log_probabilities += sentence_log_probability

    perplexity = math.exp(-sum_of_sentences_log_probabilities / M)
    return perplexity


def probability_to_perplexity(m_sentences: List[List[str]], probabilities: List[float]) -> float:
    M = 0  # Amount of words in the test data
    sum_of_sentences_probabilities = 0

    for sentence, prob in zip(m_sentences, probabilities):
        M += len(sentence)  # Add the length of the current sentence
        sum_of_sentences_probabilities += prob

    perplexity = math.exp(-sum_of_sentences_probabilities / M)
    return perplexity


def liner_interpolation_model(unigram_model, bigram_model, sentence: List[str], log_it=False) -> float:
    """
    The function computes the logarithmic probability of a sentence
    :param log_it:
    :param unigram_model:
    :param bigram_model:
    :param sentence: List of tokens word
    :return: The sentence's logarithmic probability
    """
    word1 = START
    sentence_probability = 1.0
    for i in range(len(sentence)):
        word2 = sentence[i]
        if word2 in bigram_model[word1]:
            sentence_probability *= (LAMBDA_BIGRAM * bigram_model[word1][word2] + LAMBDA_UNIGRAM * model_unigram[word2])
        else:
            sentence_probability *= (LAMBDA_UNIGRAM * unigram_model[word2])

        word1 = word2
    return math.log(sentence_probability) if log_it else sentence_probability


if __name__ == '__main__':
    # Download the data to a pickle file
    # Preprocess the given sentences to lemmatize ones - as list of tokens
    preprocess_lemmas()
    sentence0 = lemmatize_text([QUESTION2_A])
    sentence1 = lemmatize_text([QUESTION3_A])
    sentence2 = lemmatize_text([QUESTION3_B])
    # Question 1:

    model_unigram = train_unigram_model(False)
    model_bigram = train_bigram_model(False, model_unigram)

    # Question 2:
    print("*** Question 2 *** \n"
          "The predicted sentence using bigram model is: ")
    next_word = bigram_model_predict_next_word(model_bigram, sentence0)
    print(QUESTION2_A, next_word, sep=" ")

    # Question 3, using the bigram model:
    # Question 3 (a) - The probability of the following two sentences:
    print("\n*** Question 3 *** , using the bigram model:\n"
          "Question 3 (a) - The probability of the following two sentences: ")

    question3a_prob = bigram_model_compute_sentence_probability(model_bigram, sentence1, True)
    question3b_prob = bigram_model_compute_sentence_probability(model_bigram, sentence2, True)
    print("Probability for question 3a: ", round(question3a_prob, 3))
    print("Probability for question 3b: ", round(question3b_prob, 3))

    # Question 3 (b) - The perplexity of both the following two sentences:
    print("Question 3 (b) - The perplexity of both the following two sentences:")
    perplexity = bigram_model_compute_sentence_perplexity(model_bigram, [sentence1, sentence2])
    print("Perplexity for question 3b: ", round(perplexity, 3))

    # Question 4 - Estimate a new model using linear interpolation smoothing
    # between the bigram model and unigram model
    sen1_prob = liner_interpolation_model(model_unigram, model_bigram, sentence1, True)
    sen2_prob = liner_interpolation_model(model_unigram, model_bigram, sentence2, True)
    sen1_sen2_perplexity = (probability_to_perplexity
                            ([sentence1, sentence2], [sen1_prob, sen2_prob]))

    print("\n*** Question 4 *** - Estimate a new model using linear interpolation smoothing \n"
          "between the bigram model and unigram model:")
    print("The probability of 1th sentence is: ", round(sen1_prob, 3))
    print("The probability of 2nd sentence is: ", round(sen2_prob, 3))
    print("The perplexity of sentences 1 and 2 is: ", round(sen1_sen2_perplexity, 3))
