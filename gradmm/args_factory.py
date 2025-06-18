import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args(argv=None):
    """Parse arguments.

    Returns:
    """
    parser = argparse.ArgumentParser(description='LAMP attack')

    # Method and setting
    parser.add_argument('--rng_seed', type=int, default=42)
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='use baseline defaults + disable all new improvements',
    )
    parser.add_argument(
        '--dataset',
        choices=[
            'sst2',
            'rotten_tomatoes',
            'TwitterEmotion',
            'imdb',
            'rtpolarity',
        ],
        required=True,
    )
    parser.add_argument('--split', required=True)
    parser.add_argument(
        '--data_loader', choices=['batch', 'cluster'], default='batch'
    )
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument(
        '--loss', choices=['cos', 'dlg', 'tag'], default='cos'
    )
    parser.add_argument(
        '--embed_loss',
        choices=['cos', 'dlg', 'tag', 'cos_mapped_embeds'],
        default='dlg',
    )
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    # Frozen params
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--opt_alg',
        choices=['adam', 'bfgs', 'bert-adam', 'admm', 'admm_sgd'],
        default='admm',
    )
    parser.add_argument('--n_steps', type=int, default=30)   #
    parser.add_argument('--init_candidates', type=int, default=500)   #
    parser.add_argument(
        '--init',
        choices=['real_first', 'real_closest', 'random_normal', 'random_embed'],
        default='random_normal',
    )
    parser.add_argument('--init_size', type=float, default=1.4)   #
    parser.add_argument(
        '--lr_decay_type',
        type=str,
        default='StepLR',
        choices=['StepLR', 'LambdaLR'],
    )

    # Tuneable params
    # Ours:             coeff_preplexity, coeff_reg, lr, lr_decay
    # Baselines:      lr, lr_decay, tag_factor
    parser.add_argument('--coeff_perplexity', type=float, default=0.0)   #
    parser.add_argument('--coeff_reg', type=float, default=0.0)   #
    parser.add_argument(
        '--lr', type=float, default=0.008
    )   # TAG best: 0.1 (for admm 0.008)
    parser.add_argument('--lr_decay', type=float, default=0.9)   # TAG best: 0.985
    parser.add_argument(
        '--admm_rho', type=float, default=0.7
    )   # Possible range 10 - 0.001
    parser.add_argument(
        '--admm_inner_steps', type=int, default=10
    )   # Possible range 10 - 200
    parser.add_argument(
        '--tag_factor', type=float, default=None
    )   # TAG best: 1e-3
    parser.add_argument(
        '--grad_clip', type=float, default=None
    )   # TAG best: 1, ours 0.5, only applicable to BERT_Large
    parser.add_argument('--lr_max_it', type=int, default=None)

    # Debug params
    parser.add_argument('--print_every', type=int, default=10)

    # addtional params for data generation
    parser.add_argument('--model_name', type=str, default='phi')
    parser.add_argument('--n_gen', type=int, default=10)
    parser.add_argument(
        '--gen_max_tokens', type=int, default=30
    )   # include prompt length
    parser.add_argument(
        '--use_auto_gen_tokens',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--gen_grad_clip',
        type=str,
        default='',
        help='grad clip for calculating matching gradient',
    )   # "": no clip, "norm": clip by norm, "elem": clip by element -1, 1
    parser.add_argument(
        '--gen_bs', type=int, default=1
    )   # number of embeddings to generate per time
    parser.add_argument('--n_gen_samples', type=int, default=1000)
    parser.add_argument('--subset_size', type=int, default=100)
    parser.add_argument('--n_fewshot', type=int, default=0)
    parser.add_argument(
        '--work_base_dir',
        type=str,
        default='./synthetic_data/',
    )
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--topk', type=int, default=50)

    parser.add_argument(
        '--overwrite',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument(
        '--drop_non_english_tokens',
        type=str2bool,
        nargs='?',
        default=False,
    )
    parser.add_argument(
        '--use_sample_tokens_only',
        type=str2bool,
        nargs='?',
        default=False,
    )
    parser.add_argument(
        '--use_topk',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--independent_gen',
        type=str2bool,
        nargs='?',
        default=True,
    )
    parser.add_argument(
        '--print_full',
        type=str2bool,
        nargs='?',
        default=True,
    )
    parser.add_argument(
        '--include_prefix',
        type=str2bool,
        nargs='?',
        default=False,
    )
    parser.add_argument(
        '--prefix_option', choices=['fixed', 'random'], default='fixed'
    )
    parser.add_argument(
        '--conversion_method', choices=['proj', 'topk', 'concat'], default='topk'
    )
    parser.add_argument('--n_prefix', type=int, default=1)
    parser.add_argument(
        '--reg_loss_type', choices=['norm', 'embed'], default='norm'
    )
    parser.add_argument(
        '--last_layer_gradient',
        type=str2bool,
        nargs='?',
        default=True,
    )
    parser.add_argument('--skip_first_samples', type=int, default=0)
    parser.add_argument(
        '--drop_change_line_characters',
        type=str2bool,
        nargs='?',
        default=True,
    )

    parser.add_argument(
        '--use_dp',
        type=str2bool,
        nargs='?',
        default=False,
    )
    parser.add_argument(
        '--dp_c',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--dp_epsilon',
        type=float,
        default=0.05,
    )
    parser.add_argument(
        '--dp_delta',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        '--save_avg_grad',
        type=str2bool,
        nargs='?',
        default=False,
    )

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv[1:])

    # if not args.neptune is None:
    #    assert not args.neptune_label is None
    #    assert len(args.neptune_label) > 0

    # Defaults above are for Ours,
    # Use different defaults if running one of the baseline methods
    if args.baseline:
        args.init_candidates = 1
        args.use_swaps = False
        args.init_size = -1
        args.coeff_perplexity = 0.0
        args.coeff_reg = 0.0

    if args.lr_max_it is None:
        args.lr_max_it = args.n_steps

    return args
