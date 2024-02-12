from datasets import load_dataset
from transformers import ElectraTokenizerFast
from datasets import load_from_disk 
import torch

def preprocess():
    # 加载数据集和分词器
    dataset = load_dataset('openwebtext')
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')

    # 定义编码函数
    def encode(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    # 使用多进程映射编码函数
    tokenized_dataset = dataset.map(encode, batched=True, num_proc=128) # 根据您的CPU调整num_proc
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

    # 保存处理后的数据集
    tokenized_dataset.save_to_disk('./data/openwebtext/tokenized_dataset')


def generate_inputs():
    tokenizer = ElectraTokenizerFast.from_pretrained(
        'google/electra-small-discriminator'
    )
    tokenized_dataset = load_from_disk('./data/openwebtext/tokenized_dataset')
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
        
    dataloader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=128)
    
    inputs = next(iter(dataloader))
    return inputs

if __name__ == '__main__':
    # preprocess()
    inputs = generate_inputs()
    print(inputs)