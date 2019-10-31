from inputter.mnist_data_manager import MNISTDataManager
from utils.config import Config

config_train = Config('../config/train.test.config.json')
data_set = MNISTDataManager(config_train)
for i in data_set.mini_batch(4):
    print(len(i[1]))
    break
