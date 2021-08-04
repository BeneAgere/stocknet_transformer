import numpy as np
import torch
from torch import nn
from torch_models import MLP, AttentiveMLP, TransformerClassifier, JointAttentionTransformer
from DataPipeTorch import DataPipe
import time
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
import logging
import optuna


def prepare_samples(batch, feature_group):
    prices = torch.tensor(batch['price_batch'])
    language_model_activations = torch.tensor(batch['language_model_activations'])

    if feature_group == 'prices':
        features = prices
    elif feature_group == 'tweets':
        features = language_model_activations
    else:
        features = torch.cat((prices, language_model_activations), dim=2)

    features = features[:, :4, :]

    labels = torch.tensor(batch['main_mv_percent_batch'] > 0.005).float().reshape((prices.shape[0], 1))
    aux_labels = torch.tensor(batch['y_batch'])

    return features, labels, aux_labels


def run_validation(model, optimizer, val_batch_generator, loss_function, feature_group, device):
    val_loss = 0
    predictions = []
    labels_agg = []

    for i, train_batch_dict in enumerate(val_batch_generator):
        features, labels, aux_labels = prepare_samples(train_batch_dict, feature_group)
        features = features.to(device)
        labels_cpu = labels
        labels = labels.to(device)

        optimizer.zero_grad()
        model.eval()
        predicted = model(features)
        if hasattr(model, 'output_seq_len') and model.output_seq_len > 1:
            predicted = predicted[:, -1]
            labels = labels.squeeze(1)

        val_loss += loss_function(predicted, labels)
        predicted_final = predicted.reshape(-1).detach().cpu().numpy().round()
        labels_final = labels_cpu.numpy()
        predictions.append(predicted_final)
        labels_agg.append(labels_final)

    all_labels = np.concatenate(labels_agg, axis=0)
    all_predictions = np.concatenate(predictions, axis=0)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    try:
        matthews_corr = matthews_corrcoef(all_labels, all_predictions)
    except:
        matthews_corr = -1.0

    return val_loss, accuracy, precision, recall, matthews_corr, f1, np.mean(all_labels), np.mean(all_predictions)


def train_model(data_generator, model_constructor=MLP, num_epochs=1, lr=1e-4, model_kwargs={}, feature_group='prices', traning_desc='MLP'):
    components = [traning_desc, feature_group, 'epochs', str(num_epochs), 'batch_128', 'bertweet']

    for key, val in model_kwargs.items():
        components.extend((str(key), str(val)))

    log_descripttion = '_'.join(components)
    log_file_name = 'log/pytorch/' + log_descripttion + '.log'
    print(log_descripttion)
    logging.basicConfig(filename=log_file_name, filemode='w', format='(%asctime)s %(message)s')

    logger = logging.getLogger(log_descripttion)

    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

    tweet_embed_dim = 768
    prices_dim = 3
    if feature_group == 'tweets':
        feature_dim = tweet_embed_dim
    elif feature_group == 'prices':
        feature_dim = prices_dim
    else:
        feature_dim = tweet_embed_dim + prices_dim

    model_kwargs['feature_dim'] = feature_dim
    use_cuda = True
    if use_cuda:
       device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
       print('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
       device = torch.device('cpu')
       print('cpu')
    model_kwargs['device'] = device

    model = model_constructor(**model_kwargs)
    model.to(device)
    loss_function = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping_best = 1e10
    early_stopping_counter = 0
    early_stopping_threshold = 2

    for epoch in range(0, num_epochs):

        logger.info(f'Starting epoch {epoch + 1}')
        print(f'Starting epoch {epoch + 1}')
        current_loss = 0.0

        train_batch_gen = data_generator.batch_gen(phase='train')

        model.train()
        for i, train_batch_dict in enumerate(train_batch_gen):
            if use_cuda:
                torch.cuda.empty_cache()
            features, labels, aux_labels = prepare_samples(train_batch_dict, feature_group)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                features = features.to(device)
                labels = labels.to(device)
                aux_labels = aux_labels.to(device)
            outputs = model(features)

            if hasattr(model, 'output_seq_len') and model.output_seq_len > 1:

                loss = torch.zeros(labels.shape[0], device=device)
                for idx in range(1, 5):
                    labels_this_step = aux_labels[:, idx, 1]
                    loss_this_step = model.aux_target_weight * loss_function(outputs[:, idx], labels_this_step) # 
                    loss += loss_this_step
                labels_this_step = loss_this_step = None
                primary_loss = loss_function(outputs[:, -1], labels.squeeze(1))
                loss += primary_loss
                loss = loss.mean()
            else:
                per_element_loss = loss_function(outputs, labels)
                loss = per_element_loss.mean()

            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if i > 0 and i % 100 == 0:
                print("Step {i} Loss: {l}".format(i=i, l=loss.item()))
            loss = None

        if epoch > 0 and epoch % 2 == 0:
            if use_cuda:
                features = labels = aux_labels = loss = None
                torch.cuda.empty_cache()
            val_batch_gen = data_generator.batch_gen(phase='dev')  # a new gen for a new epoch
            metrics = run_validation(model, optimizer, val_batch_gen, loss_function, feature_group, device)
            val_loss, accuracy, precision, recall, matthews_corr, f1, pos, pred_pos = metrics
            print('val loss {}% change from best'.format((1.0 * val_loss - early_stopping_best) / early_stopping_best))
            if val_loss < early_stopping_best:
                early_stopping_best = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            msg = 'Epoch: {e} Val Loss: {l} Accuracy: {a} Precision: {p} Recall: {r} MCC: {m} F1: {f} Pct_Positive: {pos} Pred_Pos: {pp}'
            print(msg.format(e=epoch, l=val_loss, a=accuracy, p=precision, r=recall, m=matthews_corr, f=f1, pos=pos, pp=pred_pos))
            logger.info(msg.format(e=epoch, l=val_loss, a=accuracy, p=precision, r=recall, m=matthews_corr, f=f1, pos=pos, pp=pred_pos))

        if early_stopping_counter >= early_stopping_threshold:
            print("Stopping early")
            break

    logger.info('Training process has finished.')


if __name__ == '__main__':
    print('loading data pipe and embeddings')
    data_generator = DataPipe()

    train_mlp = False
    train_residual_attention = False
    train_transformer = False
    train_aux_transformer = False
    train_joint_attention_transformer = False
    train_joint_attention_transformer_with_aux_targets = False
    run_sweep = True

    if train_mlp:
       train_model(data_generator, model_constructor=MLP, num_epochs=3, lr=1e-5,
                   model_kwargs={'embed_dim': 8, 'dropout': 0.1}, feature_group='both', traning_desc='MLP')

    if train_residual_attention:
        train_model(data_generator, model_constructor=AttentiveMLP, num_epochs=3, lr=1e-5,
                    model_kwargs={'embed_dim': 8, 'num_heads': 1, 'dropout': 0.1},
                    feature_group='both', traning_desc='AttentiveMLP')

    if train_transformer:
        train_model(data_generator, model_constructor=TransformerClassifier, num_epochs=3, lr=1e-5,
                    model_kwargs={'embed_dim': 4, 'num_heads': 1, 'dropout': 0.1, 'n_transformer_layers': 2},
                    feature_group='both', traning_desc='Transformer')

    if train_aux_transformer:
        train_model(data_generator, model_constructor=TransformerClassifier, num_epochs=3, lr=1e-5,
                    model_kwargs={'embed_dim': 4, 'num_heads': 1, 'output_seq_len': 5, 'aux_target_weight': 0.3},
                    feature_group='both', traning_desc='AuxiliaryTransformer')

    if train_joint_attention_transformer:
        train_model(data_generator, model_constructor=JointAttentionTransformer, num_epochs=3, lr=1e-5,
                    model_kwargs={'embed_dim': 256, 'num_heads': 1, 'n_transformer_layers': 1, 'output_seq_len': 1},
                    feature_group='both', traning_desc='JointAttentionTransformer')

    if train_joint_attention_transformer_with_aux_targets:
        train_model(data_generator, model_constructor=JointAttentionTransformer, num_epochs=3, lr=1e-5,
                    model_kwargs={'embed_dim': 256, 'num_heads': 1, 'n_transformer_layers': 1, 'output_seq_len': 5},
                    feature_group='both', traning_desc='AuxiliaryJointAttentionTransformer')

    if run_sweep:
        for dropout in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            for dim in (128, 256, 512, 1024, 2048, 4096):
                for num_heads in (2, 4, 8, 16):
                    for n_layers in (1, 2, 4, 6):
                        for aux_target_weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                            try:
                                train_model(data_generator, model_constructor=JointAttentionTransformer, num_epochs=3,
                                            lr=1e-5, model_kwargs={'embed_dim': dim*4, 'num_heads': num_heads,
                                            'n_transformer_layers': n_layers, 'aux_target_weight': aux_target_weight,
                                            'output_seq_len': 5}, feature_group='both',
                                            traning_desc='AuxiliaryJointAttentionTransformer')
                            except:
                                print("Failed for Aux Joint Attention Transformer with dim {}".format(dim))

                            try:
                                train_model(data_generator, model_constructor=TransformerClassifier, num_epochs=10, lr=1e-5,
                                            model_kwargs={'embed_dim': dim, 'num_heads': num_heads,
                                            'aux_target_weight': aux_target_weight, 'dropout': dropout,
                                            'output_seq_len': 5}, feature_group='both',
                                            traning_desc='AuxiliaryTransformer')
                            except:
                                print("Failed for Aux Transformer with dim {}".format(dim))


                        try:
                            train_model(data_generator, model_constructor=JointAttentionTransformer, num_epochs=50,
                                        lr=1e-5, model_kwargs={'embed_dim': dim, 'num_heads': num_heads,
                                        'dropout': dropout, 'n_transformer_layers': n_layers},
                                        feature_group='both', traning_desc='JointAttentionTransformer')
                        except:
                            print("Failed for Joint Attention Transformer with dim {}".format(dim))

                        try:
                            train_model(data_generator, model_constructor=TransformerClassifier, num_epochs=50, lr=1e-5,
                                    model_kwargs={'embed_dim': dim, 'num_heads': num_heads, 'dropout': dropout,
                                                  'n_transformer_layers': n_layers},
                                        feature_group='both', traning_desc='Transformer')
                        except:
                            print("Failed for Transformer with dim {}".format(dim))

                    try:
                        train_model(data_generator, model_constructor=AttentiveMLP, num_epochs=50, lr=1e-5,
                                    model_kwargs={'embed_dim': dim, 'num_heads': num_heads, 'dropout': dropout},
                                    feature_group='both', traning_desc='Transformer')
                    except:
                        print("Failed for Transformer with dim {}".format(dim))

                try:
                    train_model(data_generator, model_constructor=MLP, num_epochs=50, lr=1e-5,
                                model_kwargs={'embed_dim': dim, 'dropout': dropout},
                                feature_group='both', traning_desc='MLP')

                except:
                    print("Failed for AttentiveMLP with dim {}".format(dim))
