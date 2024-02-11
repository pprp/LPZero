from lpzero.model.model_loader import load_model_from_config
import yaml 
import torch 
from lpzero.operators.zc_inputs import compute_activation, compute_gradient, \
    compute_head, compute_jacobs, \
    compute_softmax, compute_weight
from lpzero.datasets.distributed_utils.data_utils import get_lm_corpus
from lpzero.runner.eval_rank_gpt2 import parse_arguments


with open("saved_logs/random_GPT2_wt103/config_0/j2/model_config.yaml", "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

# build gpt2 model 
model = load_model_from_config('hf_gpt2_flex', model_config)
model.n_token = model_config['n_token']

# build inputs for gpt2 
args = parse_arguments()

if args.dataset == "wt103":
    eval_batch_size = 16
    eval_tgt_len = 192
elif args.dataset == "lm1b":
    eval_batch_size = 16
    eval_tgt_len = 32

data = './data/wikitext/wikitext-103'
cache_dir = './data/cachedir'
vocab = 'gpt2' if 'gpt' in args.model_type else 'word'
vocab_size = 50264 if 'gpt' in args.model_type else 267736
corpus = get_lm_corpus(data, cache_dir, args.dataset, vocab, vocab_size, refresh_cache=False)
train_itr = corpus.get_iterator("train", eval_batch_size, eval_tgt_len,
                                device=args.device, mem_len=0, ext_len=0)
if args.dataset != "lm1b":
    train_iter = train_itr.get_fixlen_iter(start=0)
else:
    train_iter = train_itr

traindata = []
num_batches = 1

print('begin')
for batch, (data, target, seq_len, _) in enumerate(train_iter, start=1):
    if batch > num_batches:
        break
    
    # Ensure each input sequence in the batch does not exceed the max_seq_length
    # if data.size(1) > max_seq_length:
    #     data = data[:, :max_seq_length]
    #     target = target[:, :max_seq_length]

    traindata.append((data, target))

inputs = torch.cat([a for a, _ in traindata], dim=-1)
targets = torch.cat([b for _, b in traindata], dim=-1)
print("end")

print(f'=====> Normal forward pass')
output = model(inputs)
print(output)

print(f'=====> Compute activation')
output = compute_activation(model, inputs)
print(len(output))

print(f'=====> Compute gradient')
output = compute_gradient(model, inputs, targets)
print(len(output))

print(f'=====> Compute head')
output = compute_head(model, inputs)
print(len(output))

print(f'=====> Compute jacobians')
output = compute_jacobs(model, inputs, targets)
print(len(output))

# print(f'=====> Compute softmax')
# output = compute_softmax(model, inputs, targets)
# print(len(output))

print(f'=====> Compute weight')
output = compute_weight(model, inputs, targets)
print(len(output))
