import argparse
import os

def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--data', type=str, default='ramani')
    parser.add_argument('--TRY', action='store_true')
    parser.add_argument('--FILTER', action='store_true')
    parser.add_argument('--grid', type=str, default='')
    parser.add_argument('--remark', type=str, default='')

    parser.add_argument('--random-walk', action='store_true')

    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')

    parser.add_argument('-l', '--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 40.')

    parser.add_argument('-r', '--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('-k', '--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('-i', '--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=2,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=0.25,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('-a', '--alpha', type=float, default=0.0,
                        help='The weight of random walk -skip-gram loss. Default is ')
    parser.add_argument('--rw', type=float, default=0.01,
                        help='The weight of reconstruction of adjacency matrix loss. Default is ')
    parser.add_argument('-w', '--walk', type=str, default='',
                        help='The walk type, empty stands for normal rw')
    parser.add_argument('-d', '--diag', type=str, default='True',
                        help='Use the diag mask or not')
    parser.add_argument(
        '-f',
        '--feature',
        type=str,
        default='walk',
        help='Features used in the first step')

    args = parser.parse_args()

    if not args.random_walk:
        args.model_name = 'model_no_randomwalk'
        args.epoch = 25
    else:
        args.model_name = 'model_{}_'.format(args.data)
        args.epoch = 25
    if args.TRY:
        args.model_name = 'try' + args.model_name
        if not args.random_walk:
            args.epoch = 5
        else:
            args.epoch = 1
    # args.epoch = 1
    args.model_name += args.remark
    print(args.model_name)

    args.save_path = os.path.join(
        '../checkpoints/', args.data, args.model_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args
