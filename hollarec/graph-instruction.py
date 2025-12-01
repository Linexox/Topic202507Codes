import crslab.instruction_tuning.test as test_module
import argparse
from crslab.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='config/crs/hollarec/redial.yaml', help='config file(yaml) path')
    parser.add_argument('-g', '--gpu', type=str, default='-1',
                        help='specify GPU id(s) to use, we now support multiple GPUs. Defaults to CPU(-1).')
    parser.add_argument('-sd', '--save_data', default=False, action='store_true',
                        help='save processed dataset')
    parser.add_argument('-rd', '--restore_data', default=False, action='store_true',
                        help='restore processed dataset')
    parser.add_argument('-ss', '--save_system', default=False, action='store_true',
                        help='save trained system')
    parser.add_argument('-rs', '--restore_system', default=False, action='store_true',
                        help='restore trained system')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                        help='use valid dataset to debug your system')
    parser.add_argument('-i', '--interact', default=False, action='store_true',
                        help='interact with your system instead of training')
    parser.add_argument('-tb', '--tensorboard', default=False, action='store_true',
                        help='enable tensorboard to monitor train performance')
    args, _ = parser.parse_known_args()
    config = Config(args.config, args.gpu, args.debug)
    test_module.main(config)