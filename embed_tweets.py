#!/usr/local/bin/python
import os
import json
from datetime import datetime, timedelta
import torch
from ConfigLoader import logger, path_parser, config_model, dates, stock_symbols, vocab, vocab_size
from collections import defaultdict
import pickle

def embed_tweets():
    tweet_path = '../data/tweet/raw'
    language_model = 'roberta' # supported: bert, bertweet, roberta
    aggregation = 'mean' # suported: mean, concat

    if language_model == 'bert':
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
    elif language_model == 'bertweet':
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        model = AutoModel.from_pretrained("vinai/bertweet-base")
    elif language_model == 'roberta':
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained('roberta-base')

    output_embedding_path = 'tweet_' + language_model + '_' + aggregation + '_embeddings.pickle'
    embeddings = defaultdict(dict)
    for stock in os.listdir(tweet_path):
        stock_embeddings_by_date = {}
        for date in os.listdir(os.path.join(tweet_path, stock)):

            with open(os.path.join(tweet_path, stock, date), 'r') as tweet:
                print('embedding {s} for {d}'.format(s=stock, d=date))
                if aggregation == 'concat':
                    vals = []
                    for line in tweet:
                        msg_dict = json.loads(line)
                        vals.append((datetime.strptime(msg_dict['created_at'], "%a %b %d %H:%M:%S %z %Y"), msg_dict['text']))
                    vals = sorted(vals, key=lambda val: val[0])

                    sorted_tweets_for_day = ' [SEP] '.join([val[1] for val in vals])
                    sorted_tweets_for_day = sorted_tweets_for_day.replace("$", ' ')
                    sorted_tweets_for_day = sorted_tweets_for_day.replace("#", ' ')
                    sorted_tweets_for_day = sorted_tweets_for_day.lower()

                    tokenized = tokenizer(sorted_tweets_for_day, truncation=True, return_tensors='pt')
                    prediction = model(**tokenized)

                    if type(prediction[1].data) == torch.Tensor:
                        embeddings[stock][date] = prediction[1].data
                        stock_embeddings_by_date[date] = prediction[1].data
                    else:
                        print('failed to embed {s} for {d}'.format(s=stock, d=date))
                elif aggregation == 'mean':
                    embeddings_per_tweet = []
                    for idx, line in enumerate(tweet):
                        msg_dict = json.loads(line)
                        text = msg_dict['text'].replace("$", ' ').replace("#", ' ').lower()
                        tokenized = tokenizer(text, truncation=True, return_tensors='pt')
                        prediction = model(**tokenized)
                        if type(prediction[1].data) == torch.Tensor:
                            embeddings_per_tweet.append(prediction[1].data)
                        else:
                            print('failed to embed tweet number {i} for {s} for {d}'.format(i=idx, s=stock, d=date))

                    mean_embedding = torch.mean(torch.cat(embeddings_per_tweet, dim=0), dim=0).unsqueeze(dim=0)

                    embeddings[stock][date] = mean_embedding
                    #stock_embeddings_by_date[date] = prediction[1].data
                    print(type(embeddings[stock][date]))

        '''
        stock_output_embedding_path = 'bert_embeddings_by_stock/{}.pickle'.format(stock)
        with open(stock_output_embedding_path, 'wb') as file:
            pickle.dump(stock_embeddings_by_date, file)
        '''
    with open(output_embedding_path, 'wb') as file:
        pickle.dump(embeddings, file)


if __name__ == '__main__':
    embed_tweets()