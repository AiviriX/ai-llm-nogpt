import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import time
start_time = time.time()

with open('content/facts.txt') as f:
    data = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500
)
texts = text_splitter.split_text(data)

df = pd.DataFrame({'facts': texts})

df.head()

text = df['facts']
encoder = SentenceTransformer('paraphrase-mpnet-base-v2')
vectors = encoder.encode(texts)

vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

def search(search_text):
    search_vector = encoder.encode(search_text)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    k = index.ntotal
    distances, ann = index.search(_vector, k=k)
    results =  pd.DataFrame({'distances':distances[0], 'ann': ann[0]})
    merge = pd.merge(results, df, left_on='ann', right_index=True)
    return merge

merge=search("Who is Darpa")
for i in range(0,5):
    print(merge['facts'][i])
    print('--------------------------')

#END Search Engine
#idk what doc is, but it is private and was trained from a previously public dataset.
# We can try to retrieve a dataset from the web and apply it to the doc file here.

# question_doc = "question: {} context: {}".format('when was ai invented?', doc)
#
# question_doc = "question: {} context: {}".format('when was ai invented?', doc)
# answer = qa_a2a_generate(question,doc, movel, tokenizer, num_answers=1, num_beams=8, min_le=96, max_len=256, max_input_length=1024, device="cpu")



print("--- %s seconds ---" % (time.time() - start_time))