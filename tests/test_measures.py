from lpzero.runner.evo_search import generate_inputs
from lpzero.utils.rank_consistency import spearman, kendalltau
import pandas as pd
from tqdm import tqdm
import os 
import yaml
import json
import torch
from lpzero.predictor.predictive import find_measures
from lpzero.common import utils
import argparse
import torch.nn.functional as F
import numpy as np
from lpzero.datasets import exp_utils

from lpzero.datasets.distributed_utils.data_utils import get_lm_corpus
from lpzero.model.model_loader import load_model_from_config

configs = []
with open("./data/BERT_benchmark.json", "r") as f:
    configs = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def parse_args():
    parent_parser = argparse.ArgumentParser(
        description='PyTorch Transformer-XL Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        )

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    # if debugging from VS then use toy mode otherwise use 1 GPU/FP32 mode to be on same side
    default_config = 'dgx1_1gpu_fp32'
    if utils.is_debugging():
        default_config = 'toy'

    cfg_parser.add_argument('--config', default=default_config) # use 'dgx1_8gpu_fp16' for V100 16GB, dgx1_1gpu_fp16, default
    cfg_parser.add_argument('--config_file', default='wt103_base.yaml')

    config_args, _ = cfg_parser.parse_known_args()

    if config_args.config is not None and config_args.config_file is not None:
        config_file_path = utils.full_path(os.path.join('.', 'nlp_logs', config_args.config_file))
        with open(config_file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]['train']
    else:
        config = {}

    general = parser.add_argument_group('general setup')
    general.add_argument('--work_dir', default='~/logdir', type=str,
                         help='Directory for the results')
    general.add_argument('--experiment_name', default='mem_transformer', type=str,
                         help='Directory for the results')
    general.add_argument('--append_dataset', action='store_true',
                         help='Automatically append dataset name to work_dir')
    general.add_argument('--append_time', action='store_true',
                         help='Automatically append current time to work_dir')
    general.add_argument('--cuda', action='store_true',
                         help='Run training on a GPU using CUDA')
    general.add_argument('--fp16', action='store_true',
                         help='Run training in fp16/mixed precision')
    general.add_argument('--restart', type=str, default='',
                         help='Restart training from the saved checkpoint')
    general.add_argument('--pretrained_path', type=str, default='',
                         help='Absolute or relative pretrained model path for finetuning or QAT')
    general.add_argument('--debug', action='store_true', default=None,
                         help='Run in debug mode (do not create exp dir)')
    general.add_argument('--log_all_ranks', action='store_true',
                         help='Enable logging from all distributed ranks')
    general.add_argument('--dllog_file', type=str, default='train_log.json',
                         help='Name of the DLLogger output file')
    general.add_argument('--txtlog_file', type=str, default='train_log.log',
                         help='Name of the txt log file')
    general.add_argument('--save_all', action='store_true',
                         help='Save all checkpoints')
    general.add_argument('--no_env', action='store_false',
                         help='Do not print info on execution env')
    general.add_argument('--no_train', action='store_false', default=False,
                         help='Only generate dataset caches, no training. Can be run on without GPU.')
    general.add_argument('--no_eval', action='store_true',
                         help='Disable model evaluation')
    general.add_argument('--refresh_cache', action='store_false', default=False,
                         help='Ignores any existing cache and overwrites it with new one')
    general.add_argument('--log_interval', type=int, default=10,
                         help='Report interval')
    general.add_argument('--target_throughput', type=float, default=None,
                         help='Target training throughput (for benchmarking)')
    general.add_argument('--target_perplexity', type=float, default=None,
                         help='Target validation perplexity (for benchmarking)')
    general.add_argument('--apex_amp_opt_level', type=str, default='O2',
                         choices=['O0', 'O1', 'O2', 'O3'],
                         help='Optimization level for apex amp')
    general.add_argument('--affinity', type=str,
                         default='socket_unique_interleaved',
                         choices=['socket', 'single', 'single_unique',
                                  'socket_unique_interleaved',
                                  'socket_unique_continuous',
                                  'disabled'],
                         help='type of CPU affinity')

    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--data', type=str, default=None,
                         help='Location of the data corpus')
    general.add_argument('--cache_dir', default='cache', type=str,
                         help='Directory to store dataset cache, either absolute or relative')
    dataset.add_argument('--dataset', type=str, # set to 'wt103' through config name unless toy mode when its wt2
                         choices=['wt103', 'wt2', 'lm1b', 'enwik8', 'text8', 'web_text', 'olx_WordData20210110', 'olx_OutlookData20210917x2', 'olx_WordData20211003', 'olx_WordData20220118_S36', 'olx_RedditWA_S100'],
                         help='Dataset name')
    dataset.add_argument('--vocab', type=str, default='word', choices=['word', 'bbpe', 'gpt2'],
                         help='Type of vocabulary')
    dataset.add_argument('--vocab_size', type=int, default=None,
                         help='Size of vocabulary')

    model = parser.add_argument_group('model setup - defaults are for base model')
    model.add_argument('--model_type', default='mem_transformer', type=str,
                     choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
                     help='Which model type to use')
    model.add_argument('--n_layer', type=int, default=16,
                       help='Number of total layers')
    model.add_argument('--n_head', nargs='+', type=int, default=8,
                       help='Number of heads')
    model.add_argument('--d_head', type=int, default=-1, # will be set by d_model // n_head
                       help='Head dimension')
    model.add_argument('--d_embed', type=int, default=-1, # will be set from d_model
                       help='Embedding dimension')
    model.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    model.add_argument('--d_inner', nargs='+', type=int, default=2048,
                       help='Inner dimension in feedforward layer')
    model.add_argument('--dropout', type=float, default=0.1,
                       help='Global dropout rate')
    model.add_argument('--dropatt', type=float, default=0.0,
                       help='Attention probability dropout rate')
    model.add_argument('--pre_lnorm', action='store_true',
                       help='Apply LayerNorm to the input instead of the output')
    model.add_argument('--attn_type', type=int, default=0,
                       help='Attention type. 0 for ours, 1 for Shaw et al,'
                       '2 for Vaswani et al, 3 for Al Rfou et al.')
    model.add_argument('--not_tied', action='store_true',
                       help='Do not tie the word embedding and softmax weights')
    model.add_argument('--clamp_len', type=int, default=-1,
                       help='Use the same pos embeddings after clamp_len')
    model.add_argument('--adaptive', action='store_true',
                       help='Use adaptive softmax')
    model.add_argument('--div_val', type=int, default=1,
                       help='Dividend value for adaptive input and softmax')
    model.add_argument('--sample_softmax', type=int, default=-1,
                       help='Number of samples in sampled softmax')
    model.add_argument('--init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--emb_init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--init_range', type=float, default=0.1,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--emb_init_range', type=float, default=0.01,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--init_std', type=float, default=0.02,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--proj_init_std', type=float, default=0.01,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--primer_square', action='store_true',
                       help='Use Primer EZ arch modifications (squared relu)')
    model.add_argument('--primer_conv', action='store_true',
                       help='Use Primer EZ arch modifications (DConv)')
    model.add_argument('--use_cache', action='store_true',
                       help='Whether to return last key/value attentions to speed decoding')
    model.add_argument('--qat', action='store_true',
                       help='Whether to perform Quantization Aware Training (usually based on pretrained model)')

    opt = parser.add_argument_group('optimizer setup')
    opt.add_argument('--optim', default='jitlamb', type=str,
                     choices=['adam', 'sgd', 'adagrad', 'lamb', 'jitlamb'],
                     help='Optimizer to use')
    opt.add_argument('--lr', type=float, default=0.01,
                     help='Initial learning rate')
    opt.add_argument('--mom', type=float, default=0.0,
                     help='Momentum for sgd')
    opt.add_argument('--scheduler', default='cosine', type=str,
                     choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'cyclic_cosine'],
                     help='LR scheduler to use')
    opt.add_argument('--scheduler_qat', default='cosine', type=str,
                     choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'cyclic_cosine'],
                     help='LR scheduler to use during QAT')
    opt.add_argument('--max_step_scheduler', type=int, default=None,
                     help='Max number of training steps for LR scheduler')
    opt.add_argument('--warmup_step', type=int, default=1000,
                     help='Number of iterations for LR warmup')
    opt.add_argument('--decay_rate', type=float, default=0.5,
                     help='Decay factor when ReduceLROnPlateau is used')
    opt.add_argument('--lr_min', type=float, default=0.0,
                     help='Minimum learning rate during annealing')
    opt.add_argument('--clip', type=float, default=0.25,
                     help='Gradient clipping')
    opt.add_argument('--weight_decay', type=float, default=0.0,
                     help='Weight decay for adam|lamb')
    opt.add_argument('--clip_nonemb', action='store_true',
                     help='Only clip the gradient of non-embedding params')
    opt.add_argument('--patience', type=int, default=0,
                     help='Patience')
    opt.add_argument('--eta_min', type=float, default=0.001,
                     help='Min learning rate for cosine scheduler')
    opt.add_argument('--mixed_qat', action='store_true',
                     help='Only clip the gradient of non-embedding params')

    training = parser.add_argument_group('training setup')
    training.add_argument('--max_step', type=int, default=40000,
                          help='Max number of training steps')
    training.add_argument('--batch_size', type=int, default=256,
                          help='Global batch size')
    training.add_argument('--local_batch_size', type=int, default=None,
                          help='Local (per-device) batch size, this setting \
                          overrides global --batch_size and sets batch_size \
                          to local_batch_size * world_size')
    training.add_argument('--batch_chunk', type=int, default=1,
                          help='Split batch into chunks and train with '
                          'gradient accumulation. 16GB V100 FP16 requires 1 chunk, FP32 requires 2 chunks')
    training.add_argument('--roll', action='store_true',
                          help='Enable random shifts within each data stream')
    training.add_argument('--tgt_len', type=int, default=192,
                          help='Number of tokens to predict')
    training.add_argument('--ext_len', type=int, default=0,
                          help='Length of the extended context')
    training.add_argument('--mem_len', type=int, default=192,
                          help='Length of the retained previous heads, number of tokens cached from previous iterations during training')
    training.add_argument('--seed', type=int, default=42,
                          help='Random seed')
    training.add_argument('--multi_gpu', default=None, type=str,
                          choices=['ddp', 'dp'],
                          help='Use multiple GPU')
    training.add_argument('--gpu0_bsz', type=int, default=-1,
                          help='Batch size on gpu 0 (for "dp" backend)')
    training.add_argument('--same_length', action='store_true',
                          help='Use the same attn length for all tokens')
    training.add_argument('--varlen', action='store_true',
                          help='Use variable length')
    training.add_argument('--swap_mem', action='store_true',
                          help='Swap memory tensors to cpu')

    val = parser.add_argument_group('validation setup')
    val.add_argument('--eval_tgt_len', type=int, default=192,
                     help='Number of tokens to predict for evaluation')
    val.add_argument('--eval_batch_size', type=int, default=16,
                     help='Eval batch size')
    val.add_argument('--eval_max_steps', type=int, default=-1,
                     help='Max eval steps')
    val.add_argument('--eval_interval', type=int, default=5000,
                     help='Evaluation interval')

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank',  type=int,
                      default=os.getenv('LOCAL_RANK', 0),
                      help='Used for multi-process training.')

    post = parser.add_argument_group('post-processing setup')
    post.add_argument('--dynamic_quantization', action='store_true',
                      help='Dynamic quantization')
    post.add_argument('--post_qat', action='store_true',
                      help='Perform QAT after training the model')

    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()

    args.tied = not args.not_tied

    if args.ext_len < 0:
        raise RuntimeError('Extended context length must be non-negative')

    # default mem_len==192, eval_tgt_len==192, tgt_len==192
    if args.mem_len == 0:
        if args.eval_tgt_len > args.ext_len + args.tgt_len:
            raise RuntimeError('eval_tgt_len should be <= tgt_len + ext_len; '
                               f'eval_tgt_len: {args.eval_tgt_len}, '
                               f'tgt_len: {args.tgt_len}, '
                               f'ext_len: {args.ext_len}')
    else:
        if args.eval_tgt_len > args.mem_len + args.tgt_len:
            raise RuntimeError('eval_tgt_len should be <= tgt_len + mem_len; '
                               f'eval_tgt_len: {args.eval_tgt_len}, '
                               f'tgt_len: {args.tgt_len}, '
                               f'mem_len: {args.mem_len}')

    if args.batch_size % args.batch_chunk != 0:
        raise RuntimeError('Batch size needs to be divisible by batch chunk')

    if args.debug is None:
        args.debug = utils.is_debugging()

    if args.eta_min == -1.0:
        args.eta_min = args.lr / 10

    args.config = config_args.config

    return args

def init():
    exp_utils.script_init()

    args = parse_args()
    
    # Initialize device and distributed backend
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda' if args.cuda else 'cpu')

    args.data, args.work_dir, args.pretrained_path, args.cache_dir, args.dataroot = \
        exp_utils.get_create_dirs(args.data, args.dataset, args.experiment_name,
                                  args.work_dir, args.pretrained_path, args.cache_dir)

    log_file = args.txtlog_file
    dllog_file = args.dllog_file
    log_file = os.path.join(args.work_dir, log_file)
    dllog_file = os.path.join(args.work_dir, dllog_file)

    # if args.debug:
    #     log_file = os.devnull
    #     dllog_file = os.devnull

    exp_utils.setup_logging(log_all_ranks=args.log_all_ranks, filename=log_file)

    # if args.config == 'toy':
    #     logging.warning('Running in toy mode which means wt2 dataset, only one step training, a lot of batch chunking for laptop GPU')

    print(args)
    
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('=' * 100)
    for k, v in args.__dict__.items():
        print('    - {} : {}'.format(k, v))
    print('=' * 100)
    return args, device

def load_data(args, device, get_file_stats=True):
    print('Generating/loading dataset...')
    corpus = get_lm_corpus(args.data, args.cache_dir, args.dataset, args.vocab,
                           vocab_size=args.vocab_size, refresh_cache=args.refresh_cache)

    if args.mem_len == 0: # default is 192
        eval_mem_len = 0
    else:
        eval_mem_len = args.mem_len + args.tgt_len - args.eval_tgt_len

    train_itr = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    valid_itr = corpus.get_iterator('valid', args.eval_batch_size,
                                  args.eval_tgt_len, device=device,
                                  mem_len=eval_mem_len, ext_len=args.ext_len)
    test_itr = corpus.get_iterator('test', args.eval_batch_size,
                                  args.eval_tgt_len, device=device,
                                  mem_len=eval_mem_len, ext_len=args.ext_len)

    file_stats = None
    if get_file_stats:
        file_stats = corpus.file_stats()
        for file_stat in file_stats:
            print(file_stat)

    return  corpus.vocab, train_itr, valid_itr, test_itr, file_stats


def create_or_load_model(args, device, ntokens)->Tuple[ArchaiModel, dict]:
    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [] # head cluster projection is never tied with embeddings
    if args.adaptive:
        assert args.dataset in ['wt103', 'wt2', 'lm1b', 'web_text'] or args.dataset.startswith('olx_')
        if args.dataset in ['wt103', 'wt2', 'web_text'] or args.dataset.startswith('olx_'):
            cutoffs = [19997, 39997, 199997, ntokens]
            tie_projs = [False] + [True] * (len(cutoffs)-1)
        elif args.dataset == 'lm1b':
            cutoffs = [59997, 99997, 639997, ntokens]
            tie_projs = [False] + [False] * (len(cutoffs)-1)
        else:
            raise RuntimeError(f'Dataset {args.dataset} not supported for set cutoffs and tie_projs')

    model_config = {
        'n_token': ntokens,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'd_model': args.d_model,
        'd_head': args.d_head,
        'd_inner': args.d_inner,
        'dropout': args.dropout,
        'dropatt': args.dropatt,
        'dtype': None,
        'tie_weight': args.tied,
        'd_embed': args.d_embed,
        'div_val': args.div_val,
        'tie_projs': tie_projs,
        'pre_lnorm': args.pre_lnorm,
        'tgt_len': args.tgt_len,
        'ext_len': args.ext_len,
        'mem_len': args.mem_len,
        'cutoffs': cutoffs,
        'adaptive': args.adaptive,
        'same_length': args.same_length,
        'attn_type': args.attn_type,
        'clamp_len': args.clamp_len,
        'sample_softmax': args.sample_softmax,

        'weight_init_type': args.init,
        'weight_init_range': args.init_range,
        'weight_init_std': args.init_std,
        'proj_init_std': args.proj_init_std,

        'primer_square': args.primer_square,
        'primer_conv': args.primer_conv,
        'use_cache': args.use_cache
        }

    model = load_model_from_config(args.model_type, model_config)

    n_params = model.get_params()
    n_all_param = n_params['total']
    n_nonemb_param = n_params['non_embedding']
    print('#params = {}'.format(n_all_param))
    print('#non emb params = {}'.format(n_nonemb_param))


    return model, model_config

def main(): 
    args, device = init()
    # load tokenizer and datasets
    vocab, train_itr, valid_itr, test_itr, file_stats = load_data(args, device)

    # create model
    ntokens = len(vocab)
    model, model_config = create_or_load_model(args, device, ntokens)

    # test forward
    inputs = generate_inputs()

    # build dataloader using lm1b
    corpus = get_lm_corpus(
        datadir='/mnt/sdb/dongpeijie/workspace/LPZero/data/wikitext/wikitext-103/',
        cachedir='./data/cachedir',
        dataset="wt103",
        vocab_type="word", # gpt2
        vocab_size=10000,
        refresh_cache=False,
    )
    train_itr = corpus.get_iterator(
        "train", 224, 192, device=device, ext_len=0
    )

    df = pd.read_csv("./BERT_results_activation.csv")


    # for i in tqdm(range(5)):
    # i = 0
    # nas_config = configs[i]["hparams"]["model_hparam_overrides"]["nas_config"]
    # config = ElectraConfig(
    #     nas_config=nas_config,
    #     num_hidden_layers=len(nas_config["encoder_layers"]),
    #     output_hidden_states=True,
    # )
    # model = ElectraModel(config)



    model.to(device)
    inputs.to(device)

    measures = find_measures(args=args, 
                             net_orig=model, 
                             dataloader=train_itr, 
                             dataload_info=["random", 224, 100],
                             device=device, 
                             loss_fn=F.cross_entropy, 
                             measure_names=["grad_norm"])

    print(measures)    

# load tokenizer and datasets
vocab, train_itr, valid_itr, test_itr, file_stats = load_data(args, device)

# create model
ntokens = len(vocab)
model, model_config = create_or_load_model(args, device, ntokens)

# test forward
inputs = generate_inputs()

# build dataloader using lm1b
corpus = get_lm_corpus(
    datadir='/mnt/sdb/dongpeijie/workspace/LPZero/data/wikitext/wikitext-103/',
    cachedir='./data/cachedir',
    dataset="wt103",
    vocab_type="word", # gpt2
    vocab_size=10000,
    refresh_cache=False,
)
train_itr = corpus.get_iterator(
    "train", 224, 192, device=device, ext_len=0
)

df = pd.read_csv("./BERT_results_activation.csv")


# for i in tqdm(range(5)):
# i = 0
# nas_config = configs[i]["hparams"]["model_hparam_overrides"]["nas_config"]
# config = ElectraConfig(
#     nas_config=nas_config,
#     num_hidden_layers=len(nas_config["encoder_layers"]),
#     output_hidden_states=True,
# )
# model = ElectraModel(config)



model.to(device)
inputs.to(device)

measures = find_measures(args=args, 
                         net_orig=model, 
                         dataloader=train_itr, 
                         dataload_info=["random", 224, 100],
                         device=device, 
                         loss_fn=F.cross_entropy, 
                         measure_names=["grad_norm"])

print(measures)