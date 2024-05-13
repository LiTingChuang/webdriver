import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from jieba import lcut
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import openai
from datetime import date
from DataRecorder import Recorder
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech

nltk.download('punkt')

def read_urls_from_txt(path):
    with open(path, "r") as file:
        urls = [line.strip() for line in file.readlines()]
    return urls

print('start')
df = pd.read_csv('/Users/ninachuang/NTUT/詳細內容All.csv')
info_df = df['內文'].to_list()
titles_df = df['標題'].to_list()
    
#docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
#dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]
print(df)

# Extract abstracts to train on and corresponding titles
# abstracts = str(info_df)
# titles = str(titles_df)

abstracts = [str(data) for data in info_df]
titles = [str(dataa) for dataa in titles_df]

print('sent_tokenize')
sentences = [sent_tokenize(abstract) for abstract in abstracts]
sentences = [sentence for doc in sentences for sentence in doc]


    # `#附加陳述==============================================
    # #KeyBERT啟發
    # #一種從 KeyBERT 工作原理中獲得靈感的方法詞類
    # #使用SpaCy的POS標記來提取單字
    # #最大邊際相關性
    # #主題詞多樣化
    # #開放人工智慧
    # #使用 ChatGPT 來標記我們的主題

    # print('keybert_model')
    # # KeyBERT
    # keybert_model = KeyBERTInspired()

    # # Part-of-Speech
    # pos_model = PartOfSpeech("en_core_web_sm")

    # # MMR
    # mmr_model = MaximalMarginalRelevance(diversity=0.3)

    # # GPT-3.5
    # prompt = """
    # I have a topic that contains the following documents:
    # [DOCUMENTS]
    # The topic is described by the following keywords: [KEYWORDS]

    # Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
    # topic: <topic label>
    # """
    # client = openai.OpenAI(api_key="sk-...")
    # openai_model = OpenAI(client, model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)

    # # All representation models
    # representation_model = {
    #     "KeyBERT": keybert_model,
    #     "OpenAI": openai_model,  # Uncomment if you will use OpenAI
    #     "MMR": mmr_model,
    #     "POS": pos_model
    # }`
#================================================================
stop_word_file_path = "jieba_stopWords.txt"

# 从txt文件读取urls
stop_word = read_urls_from_txt(stop_word_file_path)

# Pre-calculate embeddings
print('precalculate_embeddings')
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

#降維
print('umap_model')
umap_model = UMAP(n_neighbors=20, n_components=1, min_dist=0.0, metric='cosine', random_state=42)

#控制主題數量
print('hdbscan_model')
hdbscan_model = HDBSCAN(min_cluster_size=5,min_samples=15, metric='euclidean', prediction_data=True)

#資料前處理（忽略停用詞語不常用詞）
print('vectorizer_model')
vectorizer_model = CountVectorizer(stop_words = stop_word, min_df=5, ngram_range=(1, 2))

print('modeling')
topic_model = BERTopic(
    # Pipeline models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    #representation_model=representation_model,

    # Hyperparameters
    top_n_words=5,
    verbose=True

)

current_date = date.today()

print('model trensform')
topics, probs = topic_model.fit_transform(abstracts, embeddings)

print('model get_topic_info')
topic_info = topic_model.get_topic_info()
topic_info.to_csv(f"{current_date}-topic_info.csv")
print(topic_info)
print('topic_info儲存結束')

#访问单个主题
print('model get_topic')
for i in range(0,7):
     topiccc=topic_model.get_topic(i, full=True)
     print(topiccc)

topicCount=[]
topicInfo=[]
topicPoint=[]
print('model get_topics')
topiccc=topic_model.get_topics()
topic_list = topiccc.items()

print('topicCount:',topicCount)
print('topicInfo:',topicInfo)

print('model get_document_info')
document_info = topic_model.get_document_info(abstracts)
document_info.to_csv(f"{current_date}-document_info.csv")
print('document_info儲存結束')

# print('model get_topic_freq')
# topic_freq = topic_model.get_topic_freq()
# topic_freq.to_csv(f"{current_date}-topic_fre")
# print(topic_freq)
# print('topic_freq儲存結束')


print('model visualize_topics')
topic_model.visualize_topics(top_n_topics=8,custom_labels=True,width=700, height=700).show()

print('model visualize_barchart')
topic_model.visualize_barchart(top_n_topics=8,custom_labels=True ,width=300, autoscale=True).show()

print('model visualize_heatmap')
topic_model.visualize_heatmap(top_n_topics=8,width=750, height=750,custom_labels=True).show()

print('model visualize_hierarchy')
topic_model.visualize_hierarchy(top_n_topics=10,height=800, hierarchical_topics=True).show()


# print('model save')
# topic_model.save("my_model")
# print('model saveeddddddddddD')

# #自訂標籤方法========================================================
# #（自訂）標籤
# # 每個主題的預設標籤是每個主題中的前 3 個單詞，它們之間用下劃線組合。
# # 當然，這可能不是您能為某個主題想到的最佳標籤。相反，我們可以使用 .set_topic_labels 手動標記所有或某些主題。
# # 我們也可以使用 .set_topic_labels 來使用我們之前擁有的其他主題表示之一，例如 KeyBERTInspired 甚至 OpenAI。

# # Label the topics yourself
# topic_model.set_topic_labels({1: "Space Travel", 7: "Religion"})

# # or use one of the other topic representations, like KeyBERTInspired
# keybert_topic_labels = {topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()}
# topic_model.set_topic_labels(keybert_topic_labels)

# # or ChatGPT's labels
# chatgpt_topic_labels = {topic: " | ".join(list(zip(*values))[0]) for topic, values in topic_model.topic_aspects_["OpenAI"].items()}
# chatgpt_topic_labels[-1] = "Outlier Topic"
# topic_model.set_topic_labels(chatgpt_topic_labels)
# topic_model.get_topic_info()
