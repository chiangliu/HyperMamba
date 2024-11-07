import os, sys, time, json
import time
from utils import recorder
from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer
from utils import config_path_prefix
import torch
DEFAULT_RES_SAVE_PATH_PREFIX = "./res"

def train_by_param(param):
    recorder.reset()
    dataloader = HSIDataLoader(param)
    train_loader,unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset() 
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader,test_loader)
    save_dit = 'ckpt/'+ param['data']['data_sign'] +'.pt'
    torch.save(trainer.net,save_dit)
    eval_res = trainer.final_eval(test_loader)
    start_eval_time = time.time()
    end_eval_time = time.time()
    eval_time = end_eval_time - start_eval_time
    print("eval time is %s" % eval_time) 
    recorder.record_time(eval_time)
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    return recorder


include_path = [
    'IP.json',
    # 'HyperMamaba/HU.json',
    # 'HyperMamaba/WHHH.json'
]


def run_all():
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        uniq_name = param.get('uniq_name', name)
        print('start to train %s...' % uniq_name)
        train_by_param(param)
        print('model eval done of %s...' % uniq_name)
        path = '%s/%s' % (save_path_prefix, uniq_name) 
        recorder.to_file(path)

if __name__ == "__main__":
    run_all()


