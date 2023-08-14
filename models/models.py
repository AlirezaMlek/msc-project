from transformers import BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification, BertForMaskedLM
from utils.BlockNetwork import *
import torch


"""
"bert_base_uncased" refers to a specific version of BERT that has been pre-trained on a large corpus of text data, with 
a base architecture and with all text converted to lowercase. This model has 12 Transformer layers and 110 million 
parameters. The "uncased" part of the name means that the model was trained on lowercase text, and thus does not 
differentiate between capital and lowercase letters
"""
def create_bert_base_uncased():
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for param in model.parameters():
        param.requires_grad = False

    embLayer = model.bert.embeddings
    networkLayers = model.bert.encoder.layer
    outputBlock = [model.cls]


    def predictor(scores, tokenizer, mask=None):
        scores = torch.nn.Softmax(dim=2)(scores)
        indices = torch.argmax(scores, dim=2)
        if mask is None:
            return tokenizer.convert_ids_to_tokens(indices[0][1:-1],)
        else:
            n = torch.sum(mask) - 1
            return tokenizer.convert_ids_to_tokens(indices[0][1:n],)

    App = DnnApp('bert-base-uncased', 'bbu', predictor=predictor)
    return App.instantiate(tokenizer, embLayer, networkLayers, outputBlock, 768, 768)



"""
"oliverguhr/german-sentiment-bert" is a pre-trained language model based on the BERT architecture, specifically trained
 for sentiment analysis on German language text. It is a fine-tuned version of the original BERT model, which was 
 pre-trained on a large corpus of text data.
"""
def create_german_sentiment_bert():
    tokenizer = AutoTokenizer.from_pretrained("oliverguhr/german-sentiment-bert")
    model = AutoModelForSequenceClassification.from_pretrained("oliverguhr/german-sentiment-bert")

    for param in model.parameters():
        param.requires_grad = False

    embLayer = model.bert.embeddings
    networkLayers = model.bert.encoder.layer
    outputBlock = [model.bert.pooler, model.classifier, torch.nn.Softmax(dim=1)]

    def predictor(scores):
        labels = ['positive', 'negative', 'neutral']
        return labels[torch.argmax(scores)]

    App = DnnApp('german-sentiment-bert', 'gsb', predictor=predictor)
    return App.instantiate(tokenizer, embLayer, networkLayers, outputBlock, 768, 768)



"""
"oliverguhr/german-sentiment-bert" is a pre-trained language model based on the BERT architecture, specifically trained
 for sentiment analysis on German language text. It is a fine-tuned version of the original BERT model, which was 
 pre-trained on a large corpus of text data.
"""
def create_farsi_sentiment_bert():
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")
    model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")

    for param in model.parameters():
        param.requires_grad = False

    embLayer = model.bert.embeddings
    networkLayers = model.bert.encoder.layer
    outputBlock = [model.bert.pooler, model.classifier, torch.nn.Softmax(dim=1)]

    def predictor(scores):
        labels = ['HAPPY', 'SAD']
        return labels[torch.argmax(scores)]

    App = DnnApp('bert-fa-base-uncased-sentiment', 'fsb', predictor=predictor)
    return App.instantiate(tokenizer, embLayer, networkLayers, outputBlock, 768, 768)
