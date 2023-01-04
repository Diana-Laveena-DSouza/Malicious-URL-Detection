from flask import Flask, request, render_template
import pandas as pd
from utils.Dataset import Dataset
from torch.utils.data import DataLoader
from utils.transformer_model import TransformerModel
import torch
from tqdm import tqdm
import pickle
import re


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
# Function for Prediction
def predict():
    def Tokenizer(urls, vocab_dict):
        ids = []
        for url in urls:
            s = [vocab_dict[key] for key in url]
            # Add Padding if the input_len is small
            MAX_LENGTH = 256
            if len(s) < MAX_LENGTH:
                padding_len = MAX_LENGTH - len(s)
                s = s + ([0] * padding_len)
                ids.append(s)
            # Truncate, if the input_len is more than maximum length
            elif len(s) > MAX_LENGTH:
                ids.append(s[0:MAX_LENGTH])
            else:
                ids.append(s)
        return pd.Series(ids)

    vocab_dict = pickle.load(open("Token.txt", "rb"))
    most_freq_vocabs = pickle.load(open("Vocab_list.txt", "rb"))
    test_url = [x for x in request.form.values()]
    test_ = pd.concat([pd.Series(test_url), pd.Series([0])], axis=1)
    test_.columns = ['url', 'label']
    test_['url'] = [re.sub(r"https://|http://", "", url) for url in test_['url'].values]
    test_['url']=test_['url'].apply(lambda x :list(x))
    test_.reset_index(drop=True, inplace=True)
    new_url = []
    for url in test_['url']:
        for i in range(len(url)):
            if url[i] not in most_freq_vocabs:
                url[i] = '<OOV>'
            else:
                pass
    new_url.append(url)
    test_['url'] = pd.Series(new_url)
    test_['url'] = Tokenizer(test_['url'].values, vocab_dict)
    test_dataset = Dataset(url=test_.loc[:, 'url'], labels = test_.loc[:, 'label'])
    test_dataLoader = DataLoader(test_dataset, batch_size=1)
    model = TransformerModel()

    # Load the Model

    model.load_state_dict(torch.load("model/model.bin", map_location = 'cpu'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    with torch.no_grad():
        for data in tqdm(test_dataLoader, total = len(test_dataLoader)):
            token_id = data['token_id'].to(device)
            pred = model(token_id)
            if torch.argmax(pred, 1)[0] == 1:
                prediction1 = "Malicious"
            else:
               prediction1 = "Benign"
            return render_template('index.html', post_text=test_url[0], pred = prediction1)


if __name__ == "__main__":
    app.run(debug = True)