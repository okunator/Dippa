import argparse

import src.dl.lightning as lightning
from src.dl.inference.inferer import Inferer
from src.config import CONFIG


def main(conf, params):
    data_dir = params.data_dir
    dataset = params.dataset
    data_fold = params.data_fold
    stride_size = params.stride_size
    pattern_list = params.pattern_list
    fn_pattern = params.fn_pattern

    lightning_model = lightning.SegModel.from_conf(CONFIG)

    inferer = Inferer(
        model=lightning_model,
        data_dir=data_dir,
        dataset=dataset,
        data_fold=data_fold,
        stride_size=stride_size,
        fn_pattern=fn_pattern
    )

    print("Running predictions")
    inferer.run_inference()
    
    print("Running post-processing")
    inferer.post_process()
    
    print("Running instance benchmarks")
    inferer.benchmark_insts(pattern_list=pattern_list)
    
    print("Running type benchmarks")
    scores = inferer.benchmark_types(pattern_list=pattern_list)
    

if __name__ == '__main__':
    # TODO add rest of the inferrer args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help="One of ('consep', 'pannuke', 'kumar'). Defaults to None. If not Used, then dataset arg is used.",
        default=None
    )
    parser.add_argument(
        '--dataset',
        help="One of ('consep', 'pannuke', 'kumar'). Defaults to None. Used if data_dir is None. If both data_dir, and dataset are None, training data is used",
        default=None
    )
    parser.add_argument(
        '--data_fold',
        help="One of ('train', 'test'). Defaults to 'test'",
        default="test"
    )
    parser.add_argument(
        '--stride_size',
        help="Stride size for the sliding window if input images need to be patched for the network",
        default=100
    )
    parser.add_argument(
        '--fn_pattern',
        help="A regex pattern that the file names can contain. Only files containing the pattern are used.",
        default="*"
    )
    parser.add_argument(
        '--pattern_list', 
        help="List of regex patterns that the file names can contain. Averages of metrics are computed for files containing these patterns", 
        default=None
    )
    args = parser.parse_args()
    main(CONFIG, args)