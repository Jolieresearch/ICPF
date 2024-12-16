import argparse
import yaml
from pathlib import Path

def load_yaml(path):

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def build_parser(mode):
    parser = argparse.ArgumentParser()
    if mode == 'train':

        current_dir = Path(__file__).resolve().parent

        config_path = current_dir / 'config' / 'train_config.yaml'
        config = load_yaml(config_path)

        parser.add_argument('--dataset', type=str, default=config['TRAIN']['DATASET'])
        args, _ = parser.parse_known_args()
        dataset_name = args.dataset

        parser.add_argument('--seed', type=int, default=config['TRAIN']['SEED'])
        parser.add_argument('--device', type=str, default=config['TRAIN']['DEVICE'])
        parser.add_argument('--metric', type=str, default=config['TRAIN']['METRIC'])
        parser.add_argument('--load_checkpoint', type=bool, default=config['CHECKPOINT']['RESUME'])
        parser.add_argument('--checkpoint_path', type=str, default=config['CHECKPOINT']['PATH'])

        # 添加 model 相关参数
        parser.add_argument('--model_id', type=str, default=config['MODEL']['MODEL_ID'])
        parser.add_argument('--alpha', type=float, default=config['MODEL']['ALPHA'])
        parser.add_argument('--feature_num', type=int, default=config['MODEL']['FEATURE_NUM'])
        parser.add_argument('--feature_dim', type=int, default=config['MODEL']['FEATURE_DIM'])
        parser.add_argument('--label_dim', type=int, default=config['MODEL']['LABEL_DIM'])
        parser.add_argument('--retrieved_num', type=int, default=config['MODEL']['NUM_OF_RETRIEVED'])
        # parser.add_argument('--retrieval_type', type=str, default=config['MODEL']['RETRIEVAL_TYPE'])

        parser.add_argument('--vit_path', type=str, default=config['VIT']['PATH'])
        parser.add_argument('--angle_path', type=str, default=config['ANGLE']['PATH'])
        parser.add_argument('--bert_path', type=str, default=config['BERT']['PATH'])
        parser.add_argument('--blip_path', type=str, default=config['BLIP']['PATH'])
        parser.add_argument('--blip2_path', type=str, default=config['BLIP2']['PATH'])
        parser.add_argument('--dinov2_path', type=str, default=config['DINOV2']['PATH'])
        parser.add_argument('--prompt_nn_length', type=int, default=config['PROMPT']['NN_LENGTH'])
        parser.add_argument('--prompt_re_length', type=int, default=config['PROMPT']['RE_LENGTH'])

        # 添加 trainer 相关参数
        parser.add_argument('--batch_size', type=int, default=config['TRAIN']['BATCH_SIZE'])
        parser.add_argument('--epochs', type=int, default=config['TRAIN']['MAX_EPOCH'])
        parser.add_argument('--early_stop_turns', type=int, default=config['TRAIN']['EARLY_STOP_TURNS'])

        # 添加 optim 相关参数
        parser.add_argument('--optim', type=str, default=config['OPTIM']['NAME'])
        parser.add_argument('--lr', type=float, default=config['OPTIM']['LR'])

        # 添加 数据集 相关参数
        parser.add_argument('--dataset_path', type=str, default=config['DATASET'][dataset_name]['PATH'])
        parser.add_argument('--dataset_id', type=str, default=config['DATASET'][dataset_name]['DATASET_ID'])
        parser.add_argument('--frame_num', type=int, default=config['DATASET'][dataset_name]['FRAME_NUM'])
        parser.add_argument('--split', type=int, default=config['DATASET'][dataset_name]['SPLIT'])
        parser.add_argument('--frames_path', type=str, default=config['DATASET'][dataset_name]['FRAMES_PATH'])

        parser.add_argument('--save', type=str, default=config['TRAIN']['SAVE_FOLDER'])

    elif mode == 'test':

        current_dir = Path(__file__).resolve().parent

        config_path = current_dir / 'config' / 'test_config.yaml'
        config = load_yaml(config_path)

        parser.add_argument('--dataset', type=str, default=config['TEST']['DATASET'])
        args, _ = parser.parse_known_args()
        dataset_name = args.dataset

        parser.add_argument('--seed', type=int, default=config['TEST']['SEED'])
        parser.add_argument('--device', type=str, default=config['TEST']['DEVICE'])
        parser.add_argument('--save', type=str, default=config['TEST']['SAVE_FOLDER'])
        parser.add_argument('--metric', type=list, default=config['TEST']['METRIC'])
        parser.add_argument('--batch_size', type=int, default=config['TEST']['BATCH_SIZE'])

        # 添加 model 相关参数
        parser.add_argument('--model_id', type=str, default=config['MODEL']['MODEL_ID'])
        parser.add_argument('--alpha', type=float, default=config['MODEL']['ALPHA'])
        parser.add_argument('--feature_num', type=int, default=config['MODEL']['FEATURE_NUM'])
        parser.add_argument('--feature_dim', type=int, default=config['MODEL']['FEATURE_DIM'])
        parser.add_argument('--label_dim', type=int, default=config['MODEL']['LABEL_DIM'])
        parser.add_argument('--retrieved_num', type=int, default=config['MODEL']['NUM_OF_RETRIEVED'])
        # parser.add_argument('--retrieval_type', type=str, default=config['MODEL']['RETRIEVAL_TYPE'])

        parser.add_argument('--vit_path', type=str, default=config['VIT']['PATH'])
        parser.add_argument('--angle_path', type=str, default=config['ANGLE']['PATH'])
        parser.add_argument('--bert_path', type=str, default=config['BERT']['PATH'])
        parser.add_argument('--blip_path', type=str, default=config['BLIP']['PATH'])
        parser.add_argument('--blip2_path', type=str, default=config['BLIP2']['PATH'])
        parser.add_argument('--dinov2_path', type=str, default=config['DINOV2']['PATH'])
        parser.add_argument('--prompt_nn_length', type=int, default=config['PROMPT']['NN_LENGTH'])
        parser.add_argument('--prompt_re_length', type=int, default=config['PROMPT']['RE_LENGTH'])
        # 添加 optim 相关参数
        parser.add_argument('--optim', type=str, default=config['OPTIM']['NAME'])
        parser.add_argument('--lr', type=float, default=config['OPTIM']['LR'])

        # 添加 数据集 相关参数
        parser.add_argument('--dataset_path', type=str, default=config['DATASET'][dataset_name]['PATH'])
        parser.add_argument('--dataset_id', type=str, default=config['DATASET'][dataset_name]['DATASET_ID'])
        parser.add_argument('--frame_num', type=int, default=config['DATASET'][dataset_name]['FRAME_NUM'])
        parser.add_argument('--split', type=int, default=config['DATASET'][dataset_name]['SPLIT'])
        parser.add_argument('--frames_path', type=str, default=config['DATASET'][dataset_name]['FRAMES_PATH'])

        parser.add_argument('--model_path',type=str,default=config['MODEL']['TRAINED_MODEL_PATH'])

        parser.add_argument('--epochs', type=int, default=config['TEST']['MAX_EPOCH'])
        parser.add_argument('--early_stop_turns', type=int, default=config['TEST']['EARLY_STOP_TURNS'])

    return parser

if __name__ == '__main__':
    parser = build_parser('model')
    args = parser.parse_args()
    print(args)
