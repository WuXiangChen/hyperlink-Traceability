import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length=512, num_heads=8, num_layers=2, dim_feedforward=2048):
        super(TransformerDecoder, self).__init__()
        self.seq_length = seq_length

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, input_dim))

        # 解码器层
        decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出层
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, tgt, memory=None):
        # 将pooler_output扩展到序列长度
        memory = memory.unsqueeze(1).repeat(1, self.seq_length, 1)
        #添加位置编码
        memory += self.positional_encoding
        #Transformer解码
        #tgt_mask = self.transformer_decoder.generate_square_subsequent_mask(self.seq_length).to(tgt.device)
        # 然后调用 transformer_decoder
        decoded_output = self.transformer_decoder(tgt=tgt, memory=memory)
        # 输出映射到原始word embedding维度
        reconstructed_embedding = self.output_layer(decoded_output)
        return reconstructed_embedding

    def compute_loss(self, original_embedding, reconstructed_embedding):
        # 计算损失
        loss_fn = nn.MSELoss()  # 均方误差损失函数
        contains_nan_reconstructed = torch.isnan(reconstructed_embedding).any()
        contains_nan_original = torch.isnan(original_embedding).any()
        if contains_nan_reconstructed or contains_nan_original:
            print("Nan detected in the embedding")
        # 检查形状是否一致
        shapes_are_equal = reconstructed_embedding.shape == original_embedding.shape
        if not shapes_are_equal:
          print(f"The shapes are equal: {shapes_are_equal}")
        loss = loss_fn(reconstructed_embedding, original_embedding)
        return loss