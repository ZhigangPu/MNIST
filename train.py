import sys
import torch
import numpy as np
from utils.config import Config
from utils.linux_wrapper import delete_file, create_dir
from model.MLPmodel import MLPModel
from torch.nn import functional as F
from inputter.mnist_data_manager import MNISTDataManager


def evaluate_accuracy(out, label):
    """evaluate accuracy of output layer"""
    softmax_out = F.log_softmax(out, dim=1)
    predict = torch.max(softmax_out, dim=1)[1]
    correct = (predict == label).sum().item()
    total = predict.size(0)

    return correct / total


def train(config_train):
    """Train MLP model"""
    config_model = Config(config_train.config_model)
    config_train_data = Config(config_train.config_train_data)
    config_test_data = Config(config_train.config_test_data)
    batch_size = config_train.batch_size

    create_dir(config_train.save_path)
    delete_file(config_train.save_path + 'train_loss.txt')
    delete_file(config_train.save_path + 'val_loss.txt')

    train_set = MNISTDataManager(config_train_data)
    test_set = MNISTDataManager(config_test_data)
    model = MLPModel(config_model)
    device = torch.device('cuda:0' if config_train.cuda else 'cpu')
    print('use device: %s' % device)
    model.to(device)

    # initialization of weights
    uniform_unit = config_train.uniform_unit
    if np.abs(uniform_unit) > 0:
        print('uniformly initializing parameters [-%f, +%f]' % (uniform_unit, uniform_unit))
        for p in model.parameters():
            p.data.uniform_(-uniform_unit, uniform_unit)

    print('initialize optimizer, init learning rate: %f' % config_train.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr)

    epoch = iter = 0
    iter_interval = config_train.iter_interval
    valid_interval = config_train.valid_interval

    report_train_loss = cum_train_loss = 0
    val_acc_history = []
    patience = num_lr_decay = 0
    while True:
        epoch += 1

        for batch in train_set.mini_batch(batch_size):
            iter += 1
            optimizer.zero_grad()

            labels, images = batch
            out = model(images)
            labels_t = torch.tensor(labels, device=device).long()
            loss = F.cross_entropy(out, labels_t)
            report_train_loss += loss
            cum_train_loss += loss

            loss.backward()
            optimizer.step()

            if iter % iter_interval == 0:
                print('epoch {}, iter {}, avg. loss {:.2f}, acc. {:.4f}'.format(epoch, iter,
                                                              report_train_loss / iter_interval,
                                                              evaluate_accuracy(out, labels_t)
                                                              ))
                report_train_loss = 0

            if iter % valid_interval == 0:
                print('epoch, iter {}, validating ... '.format(epoch, iter))
                with torch.no_grad():
                    model.eval()
                    for batch in test_set.mini_batch(batch_size=config_train.valid_batch_size):
                        labels, images = batch
                        labels_t = torch.tensor(labels, device=device).long()
                        out = model(images)
                        acc = evaluate_accuracy(out, labels_t)
                        print('acc. {:.4f}'.format(acc))

                        loss = F.cross_entropy(out, labels_t)

                        with open(config_train.save_path + 'train_loss.txt', 'a') as f:
                            f.write(str(cum_train_loss.item() / valid_interval) + '\n')
                        with open(config_train.save_path + 'val_loss.txt', 'a') as f:
                            f.write(str(loss.item()) + '\n')
                        cum_train_loss = 0

                        is_better = len(val_acc_history) == 0 or acc > max(val_acc_history)
                        val_acc_history.append(acc)

                        if is_better:
                            patience = 0
                            print('save currently the best model to [%s]' % config_train.save_path, file=sys.stdout)
                            model.save(config_train.save_path, 'best.model')
                            torch.save(optimizer.state_dict(), config_train.save_path + 'best.optim')
                        elif patience < config_train.max_patience:
                            patience += 1
                            print('hit patience %d' % patience, file=sys.stdout)

                            if patience == config_train.max_patience:
                                num_lr_decay += 1
                                print('hit #%d trial' % num_lr_decay)
                                if num_lr_decay == config_train.max_lr_decay_num:
                                    print('eraly stop!')
                                    sys.exit(0)

                                # TODO: add more refined learning rate scheduler
                                lr = optimizer.param_groups[0]['lr'] * float(config_train.lr_decay)
                                print('load previously best model and decay learning rate to %f' % lr,
                                      file=sys.stdout)

                                # load model
                                params = torch.load(config_train.save_path + 'best.model',
                                                    map_location=lambda storage, loc: storage)
                                model.load_state_dict(params['state_dict'])
                                model = model.to(device)

                                print('restore parameters of the optimizers', file=sys.stdout)
                                optimizer.load_state_dict(torch.load(config_train.save_path + 'best.optim'))

                                # set new lr
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = lr

                                # reset patience
                                patience = 0
                        break

                model.train()

        if epoch == config_train.max_epoch:
            print('reach max epoch!')
            break


if __name__ == '__main__':
    config_train = Config('./config/train.test.config.json')
    train(config_train)
