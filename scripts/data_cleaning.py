import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Representative examples for each category
examples = {
    "what_like": ["What is the meaning?", "Which one is better?", "Who discovered this?", "Where is the place?", "When did it happen?"],
    "how_like": ["How does this work?", "In what way can this be improved?", "By what means can it be achieved?"],
    "why_like": ["Why did this occur?", "For what reason did it happen?", "How come this was done?"]
}

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

def compute_similarity(question, category_examples):
    # Combine the question with the category examples for vectorization
    all_texts = [question] + category_examples
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    # Compute cosine similarity between the question and each example
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    # Return the maximum similarity score as the category score
    return max(similarity_scores)

def weak_supervision_label(question):
    if not isinstance(question, str):
        # Error handling 
        return [0.33, 0.33, 0.33]  # Default to uncertain label
    
    question = question.lower()
    
    what_like = ["what", "which", "who", "where", "when"]
    how_like = ["how", "in what way", "by what means"]
    why_like = ["why", "for what reason", "how come"]
    
    label = np.array([0.0, 0.0, 0.0])
    
    # Define weights for the word position in the question
    weight_start = 1.5
    weight_middle = 1.0
    weight_end = 2.0
    
    # Helper function to apply weights based on position
    def apply_weights(word_list, weight):
        nonlocal label
        for word in word_list:
            if word in question:
                if question.endswith(word):
                    label += np.array([weight_end if word in what_like else 0, 
                                        weight_end if word in how_like else 0, 
                                        weight_end if word in why_like else 0])
                elif question.startswith(word):
                    label += np.array([weight_start if word in what_like else 0, 
                                        weight_start if word in how_like else 0, 
                                        weight_start if word in why_like else 0])
                else: 
                    label += np.array([weight_middle if word in what_like else 0, 
                                        weight_middle if word in how_like else 0, 
                                        weight_middle if word in why_like else 0])
    
    # Apply the weights for each category
    apply_weights(what_like, weight_middle)
    apply_weights(how_like, weight_middle)
    apply_weights(why_like, weight_middle)
    
    # Compute similarity scores for each category
    similarity_scores = np.array([
        compute_similarity(question, examples['what_like']),  # Similarity to what_like
        compute_similarity(question, examples['how_like']),   # Similarity to how_like
        compute_similarity(question, examples['why_like'])    # Similarity to why_like
    ])
    
    # Combine weights and similarity
    combined_label = label + similarity_scores
    
    # Set the highest scoring label to 1, others to 0
    max_index = np.argmax(combined_label)
    final_label = np.zeros_like(combined_label)
    final_label[max_index] = 1

    return final_label.tolist()


# Load data from CSV and JSON
data_csv = pd.read_csv('/Users/lancesanterre/pipeline_edu/data/uncleaned/q_quora.csv', low_memory=False)
data_json = pd.read_json('/Users/lancesanterre/pipeline_edu/data/uncleaned/dev-v1.1.json')

# Extract questions from JSON
questions = []
for entry in data_json["data"]:
    for paragraph in entry["paragraphs"]:
        for qa in paragraph["qas"]:
            questions.append(qa["question"])

# Add questions from CSV
questions.extend(data_csv['question1'].dropna().tolist())

# Create DataFrame
df = pd.DataFrame(questions, columns=['question'])

# Apply labeling
df['labels'] = df['question'].apply(weak_supervision_label)

# Save the DataFrame and pipeline in a single pickle file
output_file = '/Users/lancesanterre/pipeline_edu/data/processed/pipeline_and_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(df, f)

print("Pipeline and data saved successfully.")