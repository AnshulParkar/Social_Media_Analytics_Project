# STEP 0: For u is Install Libraries\
# pip install pandas numpy
# pip install nltk
# pip install gensim
# pip install matplotlib

# STEP 1: Import Libraries
import pandas as pd

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim import corpora
from gensim.models.ldamodel import LdaModel

import matplotlib.pyplot as plt

import seaborn as sns

# STEP 2: Load Dataset
data = pd.read_csv('news_data.csv')

# STEP 3: Inspect
print("\n Shape: \n",data.head(10))
print("\n Columns: \n",data.columns)
print("\n Data Types: \n",data.dtypes)
print("\n Data Descriptions: \n",data.describe())
print("\n Who contains Null value: \n",data.isnull().sum())


# STEP 4: Clean Data
# Drop duplicates
data.drop_duplicates(inplace=True)

data['Combined_Data'] = data['title'].fillna('') + ' ' + data['description'].fillna('')

print(data['Combined_Data'].head())
print(data.columns)

# Text Cleaning Function
stop_words = set(stopwords.words('english'))

def clean_text(text):  # def use karte hai function ke liya
    text = str(text).lower()
    text = re.sub(r'http\S+|www\s+', '', text)  # removes url
    text = re.sub(r'[^[a-z\s]', '', text) # Removes non-alpha
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t)>3]
    return tokens

data['tokens'] = data['Combined_Data'].apply(clean_text)
print(data['tokens'])

print(f"\nCleaned dataset shape: {data.shape}")


# STEP 5: Create Corpus for LDA
dictionary = corpora.Dictionary(data['tokens'])
print(dictionary)

corpus = [dictionary.doc2bow(tokens) for tokens in data['tokens']]

print("\n Dictionary Size: ",len(dictionary))
print("\n Corpus Size: ",len(corpus))


# STEP 6: Apply LDA Model
NUM_TOPICS = 4

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,
    passes=10,
    alpha='auto',
    per_word_topics=True
)


# STEP 7: Display Topics
print("\n" + "="*60)
print("DISCOVERED TOPICS:")
print("="*60)

for idx, topic in lda_model.print_topics(num_words=8):
    print(f"\n Topic {idx+1} : {topic}")

print("\nPrint only top words per topic (no weights) \n")
# Print only top words per topic (no weights)
for i in range(NUM_TOPICS):
    words_only = [word for word, _ in lda_model.show_topic(i, topn=8)]
    print(f"Topic {i+1}: {', '.join(words_only)}")

# STEP 8: Visualization
# Plot 1 Top words Per topics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i in range(NUM_TOPICS):
    top_words = lda_model.show_topic(i, topn=10)
    words = [w[0] for w in top_words]
    weights = [w[1] for w in top_words]

    axes[i].barh(words[::-1], weights[::-1], color=plt.cm.tab10(i / NUM_TOPICS))
    axes[i].set_title(f'Topic {i + 1}', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('Weight')
    axes[i].tick_params(axis='y', labelsize=10)

# Hide unused subplot
axes[NUM_TOPICS].axis('off')
plt.suptitle('LDA Topic Modelling — Top Words per Topic\n(Dataset: news_data.csv)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


# Bar Plot for Top 5 Words
# Get the most significant words for Topic 0
top_words = lda_model.show_topic(1,5)
words = [word for word, count in top_words]
importance = [count for word, count in top_words]

# plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=words, palette='deep')
plt.title('Top 10 Words in Topic 1')
plt.xlabel('Importance Weight')
plt.show()
