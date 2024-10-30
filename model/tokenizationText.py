import numpy as np
import torch

def encode_text_with_glove(text, glove_model_name="model/glove_preTrained/glove.6B"):
    """
    参数:
    text (str): 待编码的文本
    glove_model_name (str): GloVe 预训练模型的名称,默认为"glove-wiki-gigaword-300"

    返回:
    numpy.ndarray: 编码后的文本表示,形状为 (num_words, 300)
    """
    # 加载预训练的 GloVe 词向量模型
    wordList = np.load(glove_model_name+'/wordsList.npy')
    wordVectors = np.load(glove_model_name+'/wordVectors.npy')
    # 将文本分词
    words = text.lower().split()
    # 初始化编码向量
    encoded_text = np.zeros((len(words), 50))
    # 遍历每个词,查找其在 GloVe 模型中的向量表示
    for i, word in enumerate(words):
        if word in wordList:
            index = np.where(wordList == word)[0][0]
            encoded_text[i] = wordVectors[index]
    # 按列求和，取平均，并取其前50维度的信息
    mean_encoded_50 = np.mean(encoded_text, axis=0)
    return mean_encoded_50

def encode_NL_PL(tokenizer_embedding, NL_PL):
    tokenizer = tokenizer_embedding
    inputs = tokenizer.encode(NL_PL, return_tensors="pt", truncation=True, padding='max_length', max_length=256)
    #embed = tokenizer_embedding[1]
    #outputs = embed(inputs)
    return inputs.T
