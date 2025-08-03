import pandas as pd
import joblib
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def make_features(confit):
    print("Making features...")
    train_df = pd.read_csv(confit.data.train_csv_save_path)
    test_df = pd.read_csv(confit.data.test_csv_save_path)

# 🧮 CountVectorizer
# 	•	Converts a collection of text documents into a matrix of token counts.
# 	•	Each row is a document; each column is a word (token).
# 	•	The values are the number of times each word appears in that document.

# 🧠 TfidfVectorizer (Term Frequency–Inverse Document Frequency)
#   •	Similar to CountVectorizer, but adjusts the raw counts using TF-IDF scores.
# 	•	TF (Term Frequency): frequency of a word in a document.
# 	•	IDF (Inverse Document Frequency): down-weights common words across all documents.
# 	•	Helps reduce the influence of common but less meaningful words like “the”, “is”, etc.

    vectorizer_name = config.features.vectorizer
    vectorizer = {
        "count-vectorizer": CountVectorizer,
        "tfidf-vectorizer": TfidfVectorizer,
    }[vectorizer_name](stop_words="english") # 	stop_words="english" means the vectorizer will automatically remove common English words 

    train_input = vectorizer.fit_transform(train_df["review"])
    test_input = vectorizer.transform(test_df["review"])

    joblib.dump(train_input, config.features.train_features_save_path)
    joblib.dump(test_input, config.features.test_features_save_path)


if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    make_features(config)