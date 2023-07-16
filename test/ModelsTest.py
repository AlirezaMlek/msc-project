import unittest

import models.models as models
from utils.BlockNetwork import *

class AppTests(unittest.TestCase):

    # test app 2 : german-sentiment-bert
    def test_App2(self):
        print('\n*********** test App 2: ')
        myNetwork = BlockNetwork('myNetwork')
        DnnApp.network = myNetwork

        # create path 2
        path2, App2 = models.create_german_sentiment_bert()

        tokenizer = App2.get_input_node().tokenizer
        token = tokenize('das ist super. [SEP]', tokenizer)

        # forward data
        res = path2.forward_backup(token)
        cls = App2.predictor(res)
        self.assertEqual(cls, 'positive') # add assertion here

        token = tokenize('Mit keinem guten Ergebniss. [SEP]', tokenizer)
        res = path2.forward_backup(token)
        cls = App2.predictor(res)
        self.assertEqual(cls, 'negative') # add assertion here


    # test app 1 : bert-base-uncased
    def test_App1(self):
        print('\n*********** test App 1: ')
        myNetwork = BlockNetwork('myNetwork')
        DnnApp.network = myNetwork
        path1, App1 = models.create_bert_base_uncased()

        tokenizer = App1.get_input_node().tokenizer

        tokens = tokenize('this is a sample text sequence with a [MASK] token . [SEP]', tokenizer)

        res = path1.forward_backup(tokens)
        pred_words = App1.predictor(res, App1.get_input_node().tokenizer, tokens['attention_mask'])
        print(' '.join(pred_words))


if __name__ == '__main__':
    unittest.main()



def tokenize(data, tokenizer):
    token_data = tokenizer.encode_plus(data, return_token_type_ids=True, truncation=True, max_length=64,
                                            return_attention_mask=True, padding='max_length', return_tensors='pt'
                                            ).data

    token_data['attention_mask'] = torch.unsqueeze(token_data['attention_mask'], dim=0)

    return token_data
