#!/usr/local/bin/python
import os
import io
import json
import numpy as np
from datetime import datetime, timedelta
import random
from ConfigLoader import logger, path_parser, config_model, dates, stock_symbols, vocab, vocab_size
from transformers import AutoTokenizer, AutoModel
import torch
import pickle


class DataPipe:

    def __init__(self):
        # load path
        self.movement_path = path_parser.movement
        self.tweet_path = path_parser.preprocessed
        self.vocab_path = path_parser.vocab
        self.glove_path = path_parser.glove

        # load dates
        self.train_start_date = dates['train_start_date']
        self.train_end_date = dates['train_end_date']
        self.dev_start_date = dates['dev_start_date']
        self.dev_end_date = dates['dev_end_date']
        self.test_start_date = dates['test_start_date']
        self.test_end_date = dates['test_end_date']

        # load model config
        self.batch_size = config_model['batch_size']
        self.shuffle = config_model['shuffle']

        self.max_n_days = config_model['max_n_days']
        self.max_n_msgs = config_model['max_n_msgs']

        self.stock_embed_size = config_model['stock_embed_size']
        self.init_stock_with_word= config_model['init_stock_with_word']
        self.y_size = config_model['y_size']

        self.language_model_embed_dim = 768
        output_embedding_path = 'res/tweet_bertweet_mean_embeddings.pickle'

        with open(output_embedding_path, 'rb') as file:
            self.bert_embeddings = pickle.load(file)

        self.price_data = {}
        for stock_symbol in stock_symbols:
            stock_movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(stock_symbol))
            self.price_data[stock_symbol] = {}
            with io.open(stock_movement_path, 'r', encoding='utf8') as movement_f:
                for line in movement_f:  # descend
                    data = line.split('\t')
                    data[0] = datetime.strptime(data[0], '%Y-%m-%d').date()
                    self.price_data[stock_symbol][data[0]] = data

        self.dates_by_stock = {ss: self.price_data[ss].keys() for ss in stock_symbols if ss in self.price_data}


    @staticmethod
    def _convert_token_to_id(token, token_id_dict):
        if token not in token_id_dict:
            token = 'UNK'
        return token_id_dict[token]

    def _get_start_end_date(self, phase):
        """
            phase: train, dev, test, unit_test
            => start_date & end_date
        """
        assert phase in {'train', 'dev', 'test', 'whole', 'unit_test'}
        if phase == 'train':
            return self.train_start_date, self.train_end_date
        elif phase == 'dev':
            return self.dev_start_date, self.dev_end_date
        elif phase == 'test':
            return self.test_start_date, self.test_end_date
        elif phase == 'whole':
            return self.train_start_date, self.test_end_date
        else:
            return '2012-07-23', '2012-08-05'  # '2014-07-23', '2014-08-05'

    def _get_batch_size(self, phase):
        """
            phase: train, dev, test, unit_test
        """
        if phase == 'train':
            return self.batch_size
        elif phase == 'unit_test':
            return 5
        else:
            return 1

    def index_token(self, token_list, key='id', type='word'):
        assert key in ('id', 'token')
        assert type in ('word', 'stock')
        indexed_token_dict = dict()

        if type == 'word':
            token_list_cp = list(token_list)  # un-change the original input
            token_list_cp.insert(0, 'UNK')  # for unknown tokens
        else:
            token_list_cp = token_list

        if key == 'id':
            for id in range(len(token_list_cp)):
                indexed_token_dict[id] = token_list_cp[id]
        else:
            for id in range(len(token_list_cp)):
                indexed_token_dict[token_list_cp[id]] = id

        return indexed_token_dict

    def _get_prices_and_ts(self, ss, main_target_date):
        '''
        Generates a sample. Returns a dictionary of:
            'T': int of number of days
            'ts': list of consecutive trading days as datetime objects
            'ys': list of
            'main_mv_percent': How much the price moves for the day on which we're predicting
            'mv_percents': Bucketizes movement percentages into classes
            'prices': List of list of [high, low, close] price
        '''


        def _get_mv_class(data, use_one_hot=False):
            mv = float(data[1])
            if self.y_size == 2:
                if mv <= 1e-7:
                    return [1.0, 0.0] if use_one_hot else 0
                else:
                    return [0.0, 1.0] if use_one_hot else 1

            if self.y_size == 3:
                threshold_1, threshold_2 = -0.004, 0.005
                if mv < threshold_1:
                    return [1.0, 0.0, 0.0] if use_one_hot else 0
                elif mv < threshold_2:
                    return [0.0, 1.0, 0.0] if use_one_hot else 1
                else:
                    return [0.0, 0.0, 1.0] if use_one_hot else 2

        def _get_y(data):
            return _get_mv_class(data, use_one_hot=True)

        def _get_prices(data):
            return [float(p) for p in data[3:6]]

        def _get_mv_percents(data):
            return _get_mv_class(data)

        ts, ys, prices, mv_percents, main_mv_percent = list(), list(), list(), list(), 0.0

        if main_target_date not in self.price_data[ss]:
            return None

        date_min = main_target_date - timedelta(days=self.max_n_days-1)
        data = self.price_data[ss][main_target_date]
        ts.append(main_target_date)
        ys.append(_get_y(data))
        main_mv_percent = data[1]
        if -0.005 <= float(main_mv_percent) < 0.0055:  # discard sample with low movement percent
            return None

        for offset in range(1, self.max_n_days * 2):
            date = main_target_date - timedelta(days=offset)
            if date in self.price_data[ss]:
                data = self.price_data[ss][date]
                prices.append(_get_prices(data))  # high, low, close
                mv_percents.append(_get_mv_percents(data))

                if date >= date_min:
                    ts.append(date)
                    ys.append(_get_y(data))
                # one additional line for x_1_prices. not a referred trading day
                elif date < date_min:
                    break

        T = len(ts)
        if len(ys) != T or len(prices) != T or len(mv_percents) != T:  # ensure data legibility
            return None

        # ascend
        for item in (ts, ys, mv_percents, prices):
            item.reverse()

        prices_and_ts = {
            'T': T,
            'ts': ts,
            'ys': ys,
            'main_mv_percent': main_mv_percent,
            'mv_percents': mv_percents,
            'prices': prices,
        }

        return prices_and_ts

    def _get_unaligned_language_model_activations(self, ss, main_target_date):

        unaligned_language_model_activations = list()  # list of model activations from bert
        d_d_max = main_target_date - timedelta(days=1)
        d_d_min = main_target_date - timedelta(days=self.max_n_days)

        d = d_d_max  # descend
        while d >= d_d_min:

            datestr = datetime.strftime(d, '%Y-%m-%d')
            embeddings = None
            if datestr in self.bert_embeddings[ss]:
                embeddings = self.bert_embeddings[ss][datestr]
            unaligned_language_model_activations.append([d, embeddings])
            d -= timedelta(days=1)

        unaligned_language_model_activations.reverse()
        return unaligned_language_model_activations

    def _align_language_model_to_trading_day(self, ts, T, unaligned_language_model_activations):

        activations = torch.zeros((len(ts), self.language_model_embed_dim))
        activation_indices = []

        for activation in unaligned_language_model_activations:
            d = activation[0]
            for t in range(T):
                if d < ts[t]:
                    activation_indices.append(t)
                    break

        for i in range(len(unaligned_language_model_activations)):
            language_model_activations, t = unaligned_language_model_activations[i], activation_indices[i]

            if language_model_activations[1] is not None:
                activations[t, :] = language_model_activations[1]

        return activations

    def sample_gen_from_one_stock(self, stock_id_dict, s, phase):
        """
            generate samples for the given stock.

            => tuple, (x:dict, y_:int, price_seq: list of floats, prediction_date_str:str)
        """
        start_date, end_date = self._get_start_end_date(phase)

        main_target_dates = [date for date in self.dates_by_stock[s] if start_date <= date.isoformat() < end_date]

        if self.shuffle:  # shuffle data
            random.shuffle(main_target_dates)

        for main_target_date in main_target_dates:
            unaligned_language_model_activations = self._get_unaligned_language_model_activations(s, main_target_date)

            prices_and_ts = self._get_prices_and_ts(s, main_target_date)
            if not prices_and_ts:
                continue

            aligned_language_model_activations = self._align_language_model_to_trading_day(prices_and_ts['ts'], prices_and_ts['T'],
                                                            unaligned_language_model_activations)

            sample_dict = {
                # meta info
                'stock': self._convert_token_to_id(s, stock_id_dict),
                'main_target_date': main_target_date.isoformat(),
                'T': prices_and_ts['T'],
                # target
                'ys': prices_and_ts['ys'],
                'main_mv_percent': prices_and_ts['main_mv_percent'],
                'mv_percents': prices_and_ts['mv_percents'],
                # source
                'prices': prices_and_ts['prices'],
                'language_model_activations': aligned_language_model_activations
            }

            yield sample_dict

    def batch_gen(self, phase):
        batch_size = self._get_batch_size(phase)
        # prepare user, stock dict
        stock_id_dict = self.index_token(stock_symbols, key='token', type='stock')
        generators = [self.sample_gen_from_one_stock(stock_id_dict, s, phase) for s in stock_symbols]
        # logger.info('{0} Generators prepared...'.format(len(generators)))

        while True:
            # start_time = time.time()
            # logger.info('start to collect a batch...')
            stock_batch = np.zeros([batch_size, ], dtype=np.int32)
            T_batch = np.zeros([batch_size, ], dtype=np.int32)
            y_batch = np.zeros([batch_size, self.max_n_days, self.y_size], dtype=np.float32)
            main_mv_percent_batch = np.zeros([batch_size, ], dtype=np.float32)
            mv_percent_batch = np.zeros([batch_size, self.max_n_days], dtype=np.float32)
            price_batch = np.zeros([batch_size, self.max_n_days, 3], dtype=np.float32)
            language_model_activations_batch = np.zeros([batch_size, self.max_n_days, self.language_model_embed_dim], dtype=np.float32)

            sample_id = 0
            while sample_id < batch_size:
                gen_id = random.randint(0, len(generators)-1)
                try:
                    sample_dict = next(generators[gen_id])
                    T = sample_dict['T']
                    # meta
                    stock_batch[sample_id] = sample_dict['stock']
                    T_batch[sample_id] = T
                    # target
                    y_batch[sample_id, :T] = sample_dict['ys']
                    main_mv_percent_batch[sample_id] = sample_dict['main_mv_percent']
                    mv_percent_batch[sample_id, :T] = sample_dict['mv_percents']
                    # source
                    price_batch[sample_id, :T] = sample_dict['prices']
                    language_model_activations_batch[sample_id, :T] = sample_dict['language_model_activations']

                    sample_id += 1
                except StopIteration:
                    del generators[gen_id]
                    if generators:
                        continue
                    else:
                        return
                        #raise StopIteration

            batch_dict = {
                # meta
                'batch_size': sample_id,
                'stock_batch': stock_batch,
                'T_batch': T_batch,
                # target
                'y_batch': y_batch,
                'main_mv_percent_batch': main_mv_percent_batch,
                'mv_percent_batch': mv_percent_batch,
                # source
                'price_batch': price_batch,
                'language_model_activations': language_model_activations_batch
            }

            yield batch_dict

    def batch_gen_by_stocks(self, phase):
        batch_size = 2000
        vocab_id_dict = self.index_token(vocab, key='token')
        stock_id_dict = self.index_token(stock_symbols, key='token', type='stock')

        for s in stock_symbols:
            gen = self.sample_gen_from_one_stock(vocab_id_dict, stock_id_dict, s, phase)

            stock_batch = np.zeros([batch_size, ], dtype=np.int32)
            T_batch = np.zeros([batch_size, ], dtype=np.int32)
            y_batch = np.zeros([batch_size, self.max_n_days, self.y_size], dtype=np.float32)
            price_batch = np.zeros([batch_size, self.max_n_days, 3], dtype=np.float32)
            mv_percent_batch = np.zeros([batch_size, self.max_n_days], dtype=np.float32)
            main_mv_percent_batch = np.zeros([batch_size, ], dtype=np.float32)
            language_model_activations_batch = np.zeros([batch_size, self.language_model_embed_dim], dtype=np.float32)

            sample_id = 0
            while True:
                try:
                    sample_info_dict = next(gen)
                    T = sample_info_dict['T']

                    # meta
                    stock_batch[sample_id] = sample_info_dict['stock']
                    T_batch[sample_id] = sample_info_dict['T']
                    # target
                    y_batch[sample_id, :T] = sample_info_dict['ys']
                    main_mv_percent_batch[sample_id] = sample_info_dict['main_mv_percent']
                    mv_percent_batch[sample_id, :T] = sample_info_dict['mv_percents']
                    # source
                    price_batch[sample_id, :T] = sample_info_dict['prices']
                    language_model_activations_batch[sample_id, :T] = sample_info_dict['language_model_activations']

                    sample_id += 1
                except StopIteration:
                    break

            n_sample_threshold = 1
            if sample_id < n_sample_threshold:
                continue

            batch_dict = {
                # meta
                's': s,
                'batch_size': sample_id,
                'stock_batch': stock_batch[:sample_id],
                'T_batch': T_batch[:sample_id],
                # target
                'y_batch': y_batch[:sample_id],
                'main_mv_percent_batch': main_mv_percent_batch[:sample_id],
                'mv_percent_batch': mv_percent_batch[:sample_id],
                # source
                'price_batch': price_batch[:sample_id],
                'language_model_activations': language_model_activations_batch[:sample_id]
            }

            yield batch_dict

    def sample_mv_percents(self, phase):
        main_mv_percents = []
        for s in stock_symbols:
            start_date, end_date = self._get_start_end_date(phase)
            stock_mv_path = os.path.join(str(self.movement_path), '{}.txt'.format(s))
            main_target_dates = []

            with open(stock_mv_path, 'r') as movement_f:
                for line in movement_f:
                    data = line.split('\t')
                    main_target_date = datetime.strptime(data[0], '%Y-%m-%d').date()
                    main_target_date_str = main_target_date.isoformat()

                    if start_date <= main_target_date_str < end_date:
                        main_target_dates.append(main_target_date)

            for main_target_date in main_target_dates:
                prices_and_ts = self._get_prices_and_ts(s, main_target_date)
                if not prices_and_ts:
                    continue
                main_mv_percents.append(prices_and_ts['main_mv_percent'])

            logger.info('finished: {}'.format(s))

        return main_mv_percents
