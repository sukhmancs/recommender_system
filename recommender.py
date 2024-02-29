""" 
This is a simple article recommendation system that uses the TF-IDF algorithm 
to recommend articles to users based on the similarity of the articles. 
The program loads articles from a file, vectorizes the articles using the TfidfVectorizer, 
calculates the cosine similarity between the articles, and generates recommendations based on the similarity scores. 
The user can choose an article from the recommendations to read and get new recommendations based on the chosen article. 
The program will continue to recommend articles until the user enters an invalid choice.
"""

from csv import DictReader
from random import randint
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

NUM_RECS = 101 # number of recommendations to return to the user

def load_articles(file_path):
    """
    Loads articles from a file and returns a DataFrame. The file can be in .csv or .json format.
    
    Args:
        file_path: The path to the file to load the articles from

    Returns:
        df: The DataFrame containing the articles
    """
    # Check the file format and load the articles accordingly
    # If the file is in .csv format, use the read_csv function from pandas to load the articles
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)        
        return df
    # If the file is in .json format, use the read_json function from pandas to load the articles
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)        
        return df
    else: # If the file format is not supported, raise a ValueError
        raise ValueError("Unsupported file format. Only .csv and .json are supported.")

def init_recommendations(n, articles):
    """
    Initializes the recommendations by returning a list of n random article indices

    Args:
        n: The number of recommendations to return
        articles: The list of articles

    Returns:
        recommendations: A list of n random article indices
    """
    recommendations = [] # Initialize an empty list to store the indices of recommended articles

    # Loop n times and add a random article index to the recommendations list
    for _  in range(n):
        article = randint(0, len(articles)-1)

        # If the article is already in the recommendations list, generate a new random article index
        while article in recommendations:
            article = randint(0, len(articles)-1)
        recommendations.append(article)
    return recommendations # Return a list of n random article indices

def display_recommendations(recommendations, articles, start_index=1):
    """
    Displays recommendations. The recommendations parameter should be a list
    of index numbers representing the recommended articles.
    
    Args:
        recommendations: A list of index numbers representing the recommended articles
        articles: The list of articles

    Returns:
        None
    """    
    for i in range(len(recommendations)):
        art_num = recommendations[i]
        print(str(i+start_index)+".",articles.iloc[art_num]['title'])
    print("\n\n")

def display_article(art_num, articles, with_text=False):
    """
    Displays article 'art_num' from the articles
    
    Args:
        art_num: The index of the article to display
        articles: The list of articles
        with_text: A boolean indicating whether to print the text of the article
        
    Returns:
        None
    """
    print("\n\n")
    print("article",art_num)
    print("=========================================")
    print(articles.iloc[art_num]["title"])
    if with_text:
        print()
        print(articles.iloc[art_num]["text"])
    print("=========================================")
    print("\n\n")

def vectorize_documents(documents):
    """
    Vectorizes the documents using the TfidfVectorizer

    Args:
        documents: The list of documents to vectorize eg. ['This is the first document', 'This is the second document']

    Returns:
        tfidf_matrix: The vectorized documents  
    """
    # Initialize the TfidfVectorizer
    # The TfidfVectorizer will convert a collection of raw documents to a matrix of TF-IDF features

    # Following are some of the parameters to create a vocabulary from the documents
    # min_df=0.01 means ignore terms that appear in less than 1% of the documents    
    # max_df=0.95 means ignore terms that appear in more than 95% of the documents
    # ngram_range=(1,2) means include both unigrams and bigrams
    # analyzer='word' means that the features should be made of word n-gram 
    # other options are 'char' and 'char_wb' 
    # where 'char' creates vocabulary based on characters 
    # and 'char_wb' creates vocabulary based on characters and only considers sequences of characters within word boundaries
    # token_pattern=r'\b[a-zA-Z]{3,}\b' means that the token pattern should match words with 3 or more characters
    # preprocessor=preprocess_text is a function that preprocesses the text before tokenization
    # In this case, the preprocess_text function lemmatizes the text and removes stop words 
    # eg. 'This is the first document' -> 'first document'
    tfidf_vectorizer = TfidfVectorizer(
                                min_df=0.01, 
                                max_df=0.95, 
                                ngram_range=(1,2), 
                                analyzer='word', 
                                token_pattern=r'\b[a-zA-Z]{3,}\b')

    # Vectorize the documents using the TfidfVectorizer
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return tfidf_matrix # Return the vectorized documents

def calculate_similarity(tfidf_matrix, index):
    """
    Calculate the cosine similarity between the article at the given index and all other articles

    Args:
        tfidf_matrix: The vectorized documents
        index: The index of the article to compare to

    Returns:
        similarity_scores: The cosine similarity scores between the article at the given index and all other articles of shape (1, num_articles)
    """
    # Calculate the cosine similarity between the article at the given index and all other articles    
    similarity_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix)
    
    # Return a 2D array of shape (1, num_articles) 
    # where each element is the cosine similarity between the article at the given index and another article
    return similarity_scores

def get_recommendations(similarity_scores, articles, index, num_similar=8, num_dissimilar=2):
    """
    Generate recommendations based on the similarity scores

    Args:
        similarity_scores: The cosine similarity scores between the article at the given index and all other articles
        articles: The articles of articles
        index: The index of the article to compare to
        num_similar: The number of similar articles to recommend
        num_dissimilar: The number of dissimilar articles to recommend

    Returns:
        recommendations: A list of indices of recommended articles    
    """
    similar_recommendations = [] # Initialize an empty list to store the indices of recommended articles
    dissimilar_recommendations = [] # Initialize an empty list to store the indices of recommended articles

    # Sort the similarity scores and get the indices of the most similar and dissimilar articles
    # Argsort returns the indices that would sort the array in ascending order e.g. [1, 2, 3] -> [3, 2, 1]
    # We want the most similar articles, so we reverse the array and get the top n similar articles
    similar_indices = similarity_scores.argsort()[0][-num_similar-1:-1][::-1]
    dissimilar_indices = similarity_scores.argsort()[0][:num_dissimilar]
    
    # Loop through the indices of the most similar and dissimilar articles
    # Add the indices of the most similar and dissimilar articles to the recommendations list
    for idx in similar_indices:
        # If the article is not the same as the current article 
        # and is not already in the recommendations list, add it to the recommendations list
        if idx != index and articles['title'][idx] not in [articles['title'][i] for i in similar_recommendations]:
            similar_recommendations.append(idx)
    # Add the indices of the most dissimilar articles to the recommendations list
    for idx in dissimilar_indices:
        # If the article is not the same as the current article 
        # and is not already in the recommendations list, add it to the recommendations list
        if articles['title'][idx] not in [articles['title'][i] for i in dissimilar_recommendations]:
            dissimilar_recommendations.append(idx)

    # A list of indices of recommended articles
    # eg. [20, 30, 40, 50, 60, 70, 80, 90, 100, 110], [5, 15]
    return similar_recommendations, dissimilar_recommendations

def main():
    """
    Main function to run the article recommendation system
    """
    # Load articles from the json file 
    # articles is a DataFrame containing the articles with columns 'terms', 'title' and 'text'
    # Also remove any duplicate articles based on the title 
    # and reset the index of the DataFrame eg. from 0, 2, 3, 4 to 0, 1, 2, 3
    articles = load_articles('data/arxiv_abstracts.csv').drop_duplicates(subset='title').reset_index(drop=True)
    
    # Remove any articles with less than 6 words
    # This is to remove any articles that are too short to be useful for recommendations
    articles = articles[articles['text'].str.split().str.len() > 6]
    print("\n\n")

    # Display the first articles to choose from
    # recs is a list of indices of recommended articles eg. [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    recs = init_recommendations(NUM_RECS, articles)
    
    # Display the random recommendations as a starting point for the user to get new recommendations
    # These are some random articles that the user can choose from 
    # It will be displayed to the user once the program starts
    # The user will choose an article from the recommendations to read and get new recommendations
    print("Welcome to the article recommender system!")
    print("Here are some articles to choose from:")
    display_recommendations(recs, articles, start_index=1) 

    # Get the text column from the articles DataFrame
    # This is the column that contains the text of the articles
    # And convert it to a list of strings i.e. each row in the text column is a string in the list
    documents = articles['text'].tolist()    
    
    # Check if the matrix has already been created
    # If it has, load it from a file    
    if os.path.exists('tfidf_matrix.pkl'):
        # Load the matrix from a file        
        with open('tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
    else: # Otherwise, create the matrix and save it to a file
        print("Creating the matrix...")
        tfidf_matrix = vectorize_documents(documents)
        # Save the matrix to a file
        with open('tfidf_matrix.pkl', 'wb') as f:
            pickle.dump(tfidf_matrix, f) # Save the matrix to a file

    # Continue to recommend articles until the user enters an invalid choice
    while True:
        
        # Get user input
        # The user will choose an article from the recommendations to read and get new recommendations
        # If the user enters an invalid choice, the program will end i.e. user enters a number that is not in the list of recommendations
        # Check the type of the input
        try:
            # Get the user's choice
            choice = int(input("\nYour choice? ")) - 1
            # If the user enters an invalid choice, end the program
            if choice < 0 or choice >= len(recs):
                print("Invalid Choice. Goodbye!")
                break
        except ValueError: # If the user enters a non-integer, end the program
            print("Invalid Choice. Goodbye!")
            exit() # Exit the program
        
        # Display the chosen article
        # This is the article that the user has chosen to read
        # The user will get new recommendations based on this article
        display_article(recs[choice], articles, with_text=True)

        # Calculate similarity 
        # Calculate the similarity between the chosen article and all other articles        
        # The similarity scores will be used to find articles that are similar to the chosen article
        # And articles that are dissimilar to the chosen article
        # similarity_scores is a 2D array of shape (1, num_articles) 
        # where each element is the cosine similarity between the article at the given index and another article
        similarity_scores = calculate_similarity(tfidf_matrix, recs[choice])

        # Generate recommendations
        # Get new recommendations based on the similarity scores
        # similar_recommendations is a list of indices of recommended articles .i.e. [20, 30, 40, 50, 60, 70, 80, 90]
        # dissimilar_recommendations is a list of indices of recommended articles .i.e. [5, 15]
        similar_recommendations, dissimilar_recommendations = get_recommendations(similarity_scores, articles, recs[choice], num_similar=10, num_dissimilar=2)
        print("\nHere are some new recommendations for you:\n")

        # Display the new recommendations
        display_recommendations(similar_recommendations, articles, start_index=1)
        print("\nOr if you want something different, how about...\n")
        display_recommendations(dissimilar_recommendations, articles, start_index=len(similar_recommendations)+1)                        

        # Update the recommendations
        # Update the recommendations list with the new recommendations
        # The user will choose from these new recommendations to get new recommendations
        recs = similar_recommendations + dissimilar_recommendations

if __name__ == "__main__":
    main()
