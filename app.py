from flask import Flask, render_template, request
import pandas as pd
import openai
import numpy as np
from openai.embeddings_utils import cosine_similarity
app = Flask(__name__)



openai.api_type = "azure"
openai.api_base = "https://undp-ngd-openai-datafutures-dev-2.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "2c10778db282466e8bd61e5791b1a41b"



def prepare_search_result_description(text):
    words = text.split()[:50]
    truncated_text = ' '.join(words)
    if len(words) < len(text.split()):
        truncated_text += '...'
    return truncated_text

print("Corpus Reading...")
data_store = pd.read_csv("data/pdf_with_clean_openai_embeddings_full.csv")
data_store = data_store[["Document Title", "Link", "Content", "rephrased_content", "Page Number", "ada_embedding"]]
data_store["Page Number"] = data_store["Page Number"] + 1
data_store['Content'] = data_store['Content'].apply(prepare_search_result_description)
data_store['rephrased_content'] = data_store['rephrased_content'].apply(prepare_search_result_description)
data_store['ada_embedding'] = data_store.ada_embedding.apply(eval).apply(np.array)
data_store['ada_len'] = data_store.ada_embedding.apply(len)
print("len of all ada embedding ", set(data_store.ada_len.values.tolist()))
print("Corpus Read Successfully")


def get_embedding(text, engine="sdgi-embedding-ada-002"):
    return openai.Embedding.create(input = [text], engine=engine)['data'][0]['embedding']

def process_query(query, n=10):
    query_embedding = get_embedding(query)
    data_store['similarities'] = data_store.ada_embedding.apply(lambda x: cosine_similarity(x, query_embedding))
    res = data_store.sort_values('similarities', ascending=False).head(n)
    res = res[["Document Title", "Link", "Content", "rephrased_content", "Page Number"]]
    res = res.rename(columns={"Document Title":"title", "Content":"description", "Page Number":"page_number", "Link":"url"})
    res = res[["title", "page_number", "url", "description"]] 
    res = res.drop_duplicates().to_dict(orient='records')
    print("total number of results are ", len(res))
    return res


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        results = process_query(query)
        return render_template('index.html', results=results, query=query)
    return render_template('index.html', results=None, query=None)

if __name__ == '__main__':
    app.run(debug=True)
