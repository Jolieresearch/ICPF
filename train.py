import logging
import os
import sys
from datetime import datetime
import numpy as np
from sklearn.isotonic import spearmanr
from tqdm import tqdm
import torch
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
import random
from functools import partial

from data.dataset import MyData, custom_collate_fn
from models.ICPF import ICPF
from parsers import build_parser

BLUE = '\033[94m'
ENDC = '\033[0m'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_init(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_init_msg(logger, args):
    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")
    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")
    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")
    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")
    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")
    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")
    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")
    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")
    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")
    logger.info(BLUE + "Number of frames: " + ENDC + f"{args.frame_num}")
    logger.info(BLUE + "Number of retrieved items used in this training: " + ENDC + f"{args.retrieved_num}")
    # logger.info(BLUE + "LLM: " + ENDC + f"{args.retrieval_type}")
    logger.info(BLUE + "Length of prompt_nn: " + ENDC + f"{args.prompt_nn_length}")
    logger.info(BLUE + "Length of prompt_re: " + ENDC + f"{args.prompt_re_length}")
    logger.info(BLUE + "train dataset proportion: " + ENDC + f"{args.split}")
    logger.info(BLUE + "Save the path of checkpoint: " + ENDC + f"{args.save}")
    logger.info(BLUE + "Training Starts!" + ENDC)

def make_saving_folder_and_logger(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    script_name = os.path.basename(__file__).replace(".py", "")
    folder_name = f"{script_name}_{args.model_id}_{args.dataset_id}_{args.metric}_{timestamp}"
    father_folder_name = args.save

    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)
    os.mkdir(folder_path)
    os.mkdir(os.path.join(folder_path, "trained_model"))

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    with open(f'{father_folder_name}/{folder_name}/run_command.sh', 'w') as f:
        f.write(f'python {" ".join(sys.argv)}\n')
    with open(f'{father_folder_name}/{folder_name}/main_code.py', 'w') as f:
        with open(__file__, 'r') as code_file:
            f.write(code_file.read())

    return father_folder_name, folder_name, logger

def delete_model(father_folder_name, folder_name, min_turn):
    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}/trained_model")
    
    for i in range(len(model_name_list)):
        if model_name_list[i] != f'model_{min_turn}.pth':
            os.remove(os.path.join(f'{father_folder_name}/{folder_name}/trained_model', model_name_list[i]))

def force_stop(msg):
    print(msg)
    sys.exit(1)

def delete_special_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.replace(BLUE, '')
    content = content.replace(ENDC, '')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def train_val(args):
    # ========================================== Loading Data ===========================================
    
    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)

    device = torch.device(args.device)
    custom_collate_fn_partial = partial(custom_collate_fn)

    train_data = MyData(os.path.join(args.dataset_path, f'{args.split}/train.pkl'), args.frames_path, args.frame_num, args.retrieved_num, args.split)
    valid_data = MyData(os.path.join(args.dataset_path, 'valid.pkl'), args.frames_path, args.frame_num, args.retrieved_num, args.split)
    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate_fn_partial, num_workers=16, pin_memory=True)
    valid_data_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, collate_fn=custom_collate_fn_partial, num_workers=8)

    # ========================================== Model ===========================================
    model = ICPF(args.angle_path, args.vit_path, args.prompt_re_length, args.retrieved_num, device)
    model = model.to(device)

    loss_fn = torch.nn.MSELoss()
    loss_fn.to(device)
    optim = AdamW(model.parameters(), args.lr, weight_decay=0.01)
    
    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=1)
    model, optim, train_data_loader = accelerator.prepare(
        model, optim, train_data_loader
    )
    loss_fn.to(device)
    model = model.to(device)

    min_total_valid_loss = float('inf')
    min_turn = 0
    start_epoch = 0

    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        min_total_valid_loss = checkpoint['min_total_valid_loss']

        logger.info(f"Load checkpoint from {args.checkpoint_path} successfully!")
        logger.info(f"Resuming from epoch {start_epoch+1} with minimum validation loss {min_total_valid_loss}")

    print_init_msg(logger, args)

    for i in range(args.epochs):
        logger.info(f"-----------------------------------Epoch {i + 1} Start!-----------------------------------")

        min_train_loss, total_valid_loss = run_one_epoch(model, args.frames_path, loss_fn, optim, train_data_loader, valid_data_loader,
                                                         device, accelerator, batch_size=args.batch_size)

        logger.info(f"[ Epoch {i + 1} (train) ]: loss = {min_train_loss}")
        logger.info(f"[ Epoch {i + 1} (valid) ]: total_loss = {total_valid_loss}")

        if total_valid_loss < min_total_valid_loss:
            min_total_valid_loss = total_valid_loss
            min_turn = i + 1

        logger.critical(f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}")
        
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optim.state_dict(),
                      'min_total_valid_loss': min_total_valid_loss,
                      "epoch": i + 1}
        
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        
        path_checkpoint = f"{father_folder_name}/{folder_name}/trained_model/model_{i + 1}.pth"
        torch.save(checkpoint, path_checkpoint)
        logger.info("Model has been saved successfully!")

        if (i + 1) - min_turn > args.early_stop_turns:
            break

    delete_model(father_folder_name, folder_name, min_turn)
    logger.info(BLUE + "Training is ended!" + ENDC)
    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")

    # 加载最佳模型进行测试
    best_model_path = f"{father_folder_name}/{folder_name}/trained_model/model_{min_turn}.pth"
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_data = MyData(os.path.join(args.dataset_path, 'test.pkl'), args.frames_path, args.frame_num, args.retrieved_num, args.split)
    test_data_loader = DataLoader(dataset=test_data, batch_size=256, collate_fn=custom_collate_fn_partial, num_workers=8)
    nMSE, SRC, MAE = run_test(model, args.frames_path, loss_fn, test_data_loader, device, batch_size=args.batch_size, args=args)
    logger.info(f"[ Test Result ]:")
    logger.info(f"{args.metric[0]} = {nMSE}")
    logger.info(f"{args.metric[1]} = {SRC}")
    logger.info(f"{args.metric[2]} = {MAE}")

    with open(f'{father_folder_name}/{folder_name}/test_results.txt', 'w') as f:
        f.write(f"Test Results:\n")
        f.write(f"{args.metric[0]} = {nMSE}\n")
        f.write(f"{args.metric[1]} = {SRC}\n")
        f.write(f"{args.metric[2]} = {MAE}\n")

    logger.info("Test results have been saved.")

def run_one_epoch(model, frames_folder, loss_fn, optim, train_data_loader, valid_data_loader, device, accelerator, batch_size):
    model.train()
    min_train_loss = float('inf')
    gradient_accumulation_steps = 1
    accumulated_steps = 0

    for batch in tqdm(train_data_loader, desc='Training Progress'):
        with accelerator.accumulate(model):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            text, item_id, label, \
                retrieved_item_id_list_video, retrieved_visual_feature_embedding_video, retrieved_label_video, \
                retrieved_item_id_list_text, retrieved_text_list_text, retrieved_label_text, \
                transform_video_data = batch

            output = model.forward(text, retrieved_text_list_text, transform_video_data, retrieved_visual_feature_embedding_video, retrieved_label_video, retrieved_label_text)

            loss = loss_fn(output, label)
            accelerator.backward(loss)
            optim.step()
            optim.zero_grad()
            accumulated_steps += 1  
            if min_train_loss > loss:
                min_train_loss = loss
                
    if accumulated_steps % gradient_accumulation_steps == 0:  
        optim.step()  
        optim.zero_grad()  

    model.eval()
    total_valid_loss = 0

    with torch.no_grad():
        for batch in tqdm(valid_data_loader, desc='Validating Progress'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            text, item_id, label, \
            retrieved_item_id_list_video, retrieved_visual_feature_embedding_video, retrieved_label_video, \
            retrieved_item_id_list_text, retrieved_text_list_text, retrieved_label_text, \
            transform_video_data = batch
            output = model.forward(text, retrieved_text_list_text, transform_video_data, retrieved_visual_feature_embedding_video, retrieved_label_video, retrieved_label_text)

            output = output.to('cpu')
            label = label.to('cpu')
            output = np.array(output)
            label = np.array(label)

            MAE = mean_absolute_error(label, output)
            loss = MAE
            total_valid_loss += loss

    return min_train_loss, total_valid_loss

def run_test(model, frames_folder, loss_fn, test_data_loader, device, batch_size, args):
    model.eval()
    total_test_step = 0
    total_MAE = 0
    total_nMSE = 0
    total_SRC = 0

    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc='Testing Progress'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            text, item_id, label, \
            retrieved_item_id_list_video, retrieved_visual_feature_embedding_video, retrieved_label_video, \
            retrieved_item_id_list_text, retrieved_text_list_text, retrieved_label_text, \
            transform_video_data = batch
            output = model.forward(text, retrieved_text_list_text, transform_video_data, retrieved_visual_feature_embedding_video, retrieved_label_video, retrieved_label_text)
            
            output = output.to('cpu')
            label = label.to('cpu')
            output = np.array(output)
            label = np.array(label)

            MAE = mean_absolute_error(label, output)
            SRC, _ = spearmanr(output, label)
            nMSE = np.mean(np.square(output - label)) / (label.std() ** 2)

            total_test_step += 1
            total_MAE += MAE
            total_SRC += SRC
            total_nMSE += nMSE

    return total_nMSE / total_test_step, total_SRC / total_test_step, total_MAE / total_test_step


def main():
    parser = build_parser('train')
    args = parser.parse_args()
    seed_init(args.seed)
    train_val(args)

if __name__ == '__main__':
    main()