from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import nltk
import json

# Define any helper functions first
def load_dict_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, drop_out, bidirectional=True):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, dropout=drop_out)
        self.linear = torch.nn.Linear(2 * hidden_dim * num_layers, 43)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.linear(hidden)
        return x


app = Flask(__name__)
CORS(app)

# Load the model and other necessary components here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mod = torch.load("/Users/ycliao/CSE6242/Project/emoji_model.pth", map_location=device)
mod.eval()

# Load your vocab and label_to_emoji
vocab = load_dict_from_json('/Users/ycliao/CSE6242/Project/train_vocab.json')
label_to_emoji = load_dict_from_json('/Users/ycliao/CSE6242/Project/label_to_emoji.json')

def get_model_prediction(input_sentence):
    tokenizer = nltk.tokenize.TweetTokenizer()
    tokenized_sentence = tokenizer.tokenize(input_sentence.lower())
    ind = []
    for token in tokenized_sentence:
        if token in vocab: ind.append(vocab[token])
        else:
            if token.startswith("#"): ind.append(vocab["UNK_HASHTAG"])
            elif token.startswith("@"): ind.append(vocab["UNK_MENTION"])
            else: ind.append(vocab["UNK_WORD"])
        if len(ind) >= 50: break
    tensor = torch.LongTensor([ind]).to(device)
    output = mod.forward(tensor)
    _, output_max = torch.topk(output, k=1, dim=1)
    emoji = label_to_emoji[str(output_max[0,0].item())]
    return emoji

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_message = data['message']
    bot_reply = get_model_prediction(user_message)
    return jsonify({'reply': bot_reply})

def load_dict_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

if __name__ == '__main__':
    app.run(port=5000)
