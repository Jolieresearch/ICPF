import pandas as pd
from tqdm import tqdm


def stack_retrieved_feature(retrieval_type, train_path, valid_path, test_path, retrieved_item_num):

    df_train = pd.read_pickle(train_path)
    df_test = pd.read_pickle(test_path)
    df_valid = pd.read_pickle(valid_path)
    df_database = pd.concat([df_train, df_test, df_valid], axis=0)
    df_database.reset_index(drop=True, inplace=True)

    retrieved_visual_feature_embedding_cls_list = []
    retrieved_text_list = []
    retrieved_textual_feature_embedding_list = []
    retrieved_label_list = []

    for i in tqdm(range(len(df_train))):

        id_list = df_train[f'retrieved_item_id_list_{retrieval_type}'][i][:retrieved_item_num]

        current_retrieved_visual_feature_embedding_cls_list = []
        current_retrieved_text_list = []
        current_retrieved_textual_feature_embedding_list = []
        current_retrieved_label_list = []

        for j in range(len(id_list)):
            item_id = id_list[j]
            index = df_database[df_database['item_id'] == item_id].index[0]
            current_retrieved_visual_feature_embedding_cls_list.append(df_database['visual_feature_embedding_cls'][index])
            current_retrieved_textual_feature_embedding_list.append(df_database['textual_feature_embedding'][index])
            current_retrieved_label_list.append(df_database['label'][index])
            current_retrieved_text_list.append(df_database['text'][index])

        retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)
        retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)
        retrieved_label_list.append(current_retrieved_label_list)
        retrieved_text_list.append(current_retrieved_text_list)

    df_train[f'retrieved_visual_feature_embedding_cls_{retrieval_type}'] = retrieved_visual_feature_embedding_cls_list
    df_train[f'retrieved_textual_feature_embedding_{retrieval_type}'] = retrieved_textual_feature_embedding_list
    df_train[f'retrieved_label_{retrieval_type}'] = retrieved_label_list
    df_train[f'retrieved_text_{retrieval_type}'] = retrieved_text_list

    df_train.to_pickle(train_path)

    retrieved_visual_feature_embedding_cls_list = []
    retrieved_text_list = []
    retrieved_textual_feature_embedding_list = []
    retrieved_label_list = []

    for i in tqdm(range(len(df_test))):

        id_list = df_test[f'retrieved_item_id_list_{retrieval_type}'][i][:retrieved_item_num]

        current_retrieved_visual_feature_embedding_cls_list = []
        current_retrieved_text_list = []
        current_retrieved_textual_feature_embedding_list = []
        current_retrieved_label_list = []


        for j in range(len(id_list)):
            item_id = id_list[j]

            index = df_database[df_database['item_id'] == item_id].index[0]
            current_retrieved_visual_feature_embedding_cls_list.append(df_database['visual_feature_embedding_cls'][index])
            current_retrieved_textual_feature_embedding_list.append(df_database['textual_feature_embedding'][index])
            current_retrieved_label_list.append(df_database['label'][index])
            current_retrieved_text_list.append(df_database['text'][index])


        retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)
        retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)
        retrieved_label_list.append(current_retrieved_label_list)
        retrieved_text_list.append(current_retrieved_text_list)

    df_test[f'retrieved_visual_feature_embedding_cls_{retrieval_type}'] = retrieved_visual_feature_embedding_cls_list
    df_test[f'retrieved_textual_feature_embedding_{retrieval_type}'] = retrieved_textual_feature_embedding_list
    df_test[f'retrieved_label_{retrieval_type}'] = retrieved_label_list
    df_test[f'retrieved_text_{retrieval_type}'] = retrieved_text_list

    df_test.to_pickle(test_path)

    retrieved_visual_feature_embedding_cls_list = []
    retrieved_text_list = []
    retrieved_textual_feature_embedding_list = []
    retrieved_label_list = []

    for i in tqdm(range(len(df_valid))):

        id_list = df_valid[f'retrieved_item_id_list_{retrieval_type}'][i][:retrieved_item_num]

        current_retrieved_visual_feature_embedding_cls_list = []
        current_retrieved_text_list = []
        current_retrieved_textual_feature_embedding_list = []
        current_retrieved_label_list = []

        for j in range(len(id_list)):
            item_id = id_list[j]

            index = df_database[df_database['item_id'] == item_id].index[0]
            current_retrieved_visual_feature_embedding_cls_list.append(df_database['visual_feature_embedding_cls'][index])
            current_retrieved_textual_feature_embedding_list.append(df_database['textual_feature_embedding'][index])
            current_retrieved_label_list.append(df_database['label'][index])
            current_retrieved_text_list.append(df_database['text'][index])

        retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)
        retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)
        retrieved_label_list.append(current_retrieved_label_list)
        retrieved_text_list.append(current_retrieved_text_list)

    df_valid[f'retrieved_visual_feature_embedding_cls_{retrieval_type}'] = retrieved_visual_feature_embedding_cls_list
    df_valid[f'retrieved_textual_feature_embedding_{retrieval_type}'] = retrieved_textual_feature_embedding_list
    df_valid[f'retrieved_label_{retrieval_type}'] = retrieved_label_list
    df_valid[f'retrieved_text_{retrieval_type}'] = retrieved_text_list
    

    df_valid.to_pickle(valid_path)

if __name__ == "__main__":

    train_path = r'train.pkl'
    valid_path = r'valid.pkl'
    test_path = r'test.pkl'
    retrieval_type = "text"
    
    stack_retrieved_feature(retrieval_type, train_path, valid_path, test_path, 23)