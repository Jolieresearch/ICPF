import argparse
import os

import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torch


def compute_similarity_in_batches(query_vectors, memory_bank, memory_bank_id, batch_size):
    r_id_list = []
    sims_list = []
    for i in tqdm(range(0, len(query_vectors), batch_size)):
        batch = query_vectors[i:i + batch_size].unsqueeze(1)
        similarity = F.cosine_similarity(batch, memory_bank.unsqueeze(0), dim=-1)
        sim_scores, top_k_id = torch.topk(similarity, k=26, dim=-1)
        # print(sim_scores.shape)
        # print(top_k_id.shape)
        for j in range(batch.size(0)):
            id_index = i + j
            id = memory_bank_id[id_index] if id_index < len(memory_bank_id) else None
            retrieved_ids = [memory_bank_id[idx] for idx in top_k_id[j].tolist() if memory_bank_id[idx] != id]
            sim_score = sim_scores[j, 1:]
            # print(sim_score.shape)
            if len(retrieved_ids) > 25:
                retrieved_ids = retrieved_ids[:25]
                # sim_score = sim_score[:10]
            r_id_list.append(retrieved_ids)
            # print(len(retrieved_ids))
            sims_list.append(sim_score.tolist())
    return r_id_list, sims_list


def main(args):
    df_train = pd.read_pickle(os.path.join(args.data_path, args.dataset_id, 'train.pkl'))
    df_valid = pd.read_pickle(os.path.join(args.data_path, args.dataset_id, 'valid.pkl'))
    df_test = pd.read_pickle(os.path.join(args.data_path, args.dataset_id, 'test.pkl'))

    train_q_i = df_train['video_features'].tolist()
    train_q_t = df_train['text_features'].tolist()
    train_item_id = df_train['item_id'].tolist()

    valid_q_i = df_valid['video_features'].tolist()
    valid_q_t = df_valid['text_features'].tolist()
    valid_item_id = df_valid['item_id'].tolist()

    test_q_i = df_test['video_features'].tolist()
    test_q_t = df_test['text_features'].tolist()

    # two retrieval bank
    r_v_i = train_q_i + valid_q_i
    r_v_t = train_q_t + valid_q_t
    memory_bank_id = train_item_id + valid_item_id

    # make retrieval vectors to tensor
    r_v_i = torch.tensor(r_v_i).squeeze(1).to(args.device)
    r_v_t = torch.tensor(r_v_t).squeeze(1).to(args.device)
    test_q_i = torch.tensor(test_q_i).squeeze(1).to(args.device)
    test_q_t = torch.tensor(test_q_t).squeeze(1).to(args.device)
    train_q_i = torch.tensor(train_q_i).squeeze(1).to(args.device)
    train_q_t = torch.tensor(train_q_t).squeeze(1).to(args.device)
    valid_q_i = torch.tensor(valid_q_i).squeeze(1).to(args.device)
    valid_q_t = torch.tensor(valid_q_t).squeeze(1).to(args.device)

    # Inter-modal retrieval text to text
    df_train['retrieved_item_id_list_text'], df_train[
        'retrieved_item_similarity_list_text'] = compute_similarity_in_batches(train_q_t, r_v_t, memory_bank_id,
                                                                               args.batch_size)
    df_train['retrieved_item_id_list_video'], df_train[
        'retrieved_item_similarity_list_video'] = compute_similarity_in_batches(train_q_i, r_v_i, memory_bank_id,
                                                                                args.batch_size)
    df_valid['retrieved_item_id_list_text'], df_valid[
        'retrieved_item_similarity_list_text'] = compute_similarity_in_batches(valid_q_t, r_v_t, memory_bank_id,
                                                                               args.batch_size)
    df_valid['retrieved_item_id_list_video'], df_valid[
        'retrieved_item_similarity_list_video'] = compute_similarity_in_batches(valid_q_i, r_v_i, memory_bank_id,
                                                                                args.batch_size)
    df_test['retrieved_item_id_list_text'], df_test[
        'retrieved_item_similarity_list_text'] = compute_similarity_in_batches(test_q_t, r_v_t, memory_bank_id,
                                                                               args.batch_size)
    df_test['retrieved_item_id_list_video'], df_test[
        'retrieved_item_similarity_list_video'] = compute_similarity_in_batches(test_q_i, r_v_i, memory_bank_id,
                                                                                args.batch_size)
    print("==> Inter-modal retrieval is done!")

    # save results
    df_train.to_pickle(os.path.join(args.data_path, args.dataset_id, 'train.pkl'))
    df_valid.to_pickle(os.path.join(args.data_path, args.dataset_id, 'valid.pkl'))
    df_test.to_pickle(os.path.join(args.data_path, args.dataset_id, 'test.pkl'))
    print(f"==> Saved retrieval results for {args.dataset_id}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, default='microlens')
    parser.add_argument('--data_path', type=str, default='datasets')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(args)
