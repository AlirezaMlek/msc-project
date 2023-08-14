import unittest
from utils.PathMaker import create_new_path
from models import models
from utils.BlockNetwork import *

class MyTestCase(unittest.TestCase):
    def test_something(self):

        device = torch.device('mps')

        myNetwork = BlockNetwork('myNetwork')
        DnnApp.network = myNetwork

        path1, App1 = models.create_bert_base_uncased()
        path2, App2 = models.create_german_sentiment_bert()

        model = create_new_path('test', App1, 7, 10, App2, 8, 8)
        tokenizer = App1.get_input_node().tokenizer

        model.to(device)

        model.fetch_fc()

        tokens = tokenize('this is a sample text sequence with a [MASK] token . [SEP]', tokenizer)

        res = model.forward_backup(tokens)
        pred_words = App1.predictor(res, App1.get_input_node().tokenizer, tokens['attention_mask'])
        print(' '.join(pred_words))



def tokenize(data, tokenizer):
    token_data = tokenizer.encode_plus(data, return_token_type_ids=True, truncation=True, max_length=64,
                                            return_attention_mask=True, padding='max_length', return_tensors='pt'
                                            ).data

    token_data['attention_mask'] = torch.unsqueeze(token_data['attention_mask'], dim=0)

    return token_data


if __name__ == '__main__':
    unittest.main()
