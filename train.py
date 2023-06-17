import argparse

from utils import main_process

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--model', type=str, default='MTL',
                        help='The used model type: MTL, single_event, single_distance, multi_classifier')
    parser.add_argument('--running_mode', type=str, help='running mode: train, test')
    parser.add_argument('--GPU_device', default=True, type=bool, help='Whether to use GPU')
    parser.add_argument('--batch_size', default=32, type=int, help='The batch size for training or test')
    parser.add_argument('--epoch_num', default=40, type=int, help='The Training epoch')
    parser.add_argument('--random_state', default=1, type=int, help='The random state for dataset divison')
    parser.add_argument('--fold_index', default=0, type=int, help='The fold indedx in five-fold cross validation')
    parser.add_argument('--output_savedir', default='./', type=str, help='The saving directory for output files')
    parser.add_argument('--model_path', default='./', type=str, help='The path of saved model')
    parser.add_argument('--dataset_ram', default=True, type=bool,
                        help='Whether to put all the dataset into the memory during training')
    parser.add_argument('--trainVal_set_striking', default='./dataset/striking_train', type=str,
                        help='Path of Training and validation dataset for striking event')
    parser.add_argument('--trainVal_set_excavating', default='./dataset/excavating_train', type=str,
                        help='Path of Training and validation dataset for excavating event')
    parser.add_argument('--test_set_striking', default='./dataset/striking_test', type=str,
                        help='Path of Training and validation dataset for striking event')
    parser.add_argument('--test_set_excavating', default='./dataset/excavating_test', type=str,
                        help='Path of Training and validation dataset for excavating event')

    args = parser.parse_args()

    main_process(model_type=args.model,
                 GPU_device=args.GPU_device,
                 random_state=args.random_state,
                 fold_index=args.fold_index,
                 is_test=False,
                 pth_file=None,
                 log_savedir=args.output_savedir,
                 batch_size=args.batch_size,
                 epoch_num=args.epoch_num,
                 dataset_ram=args.dataset_ram,
                 trainVal_set_striking=args.trainVal_set_striking,
                 trainVal_set_excavating=args.trainVal_set_excavating
                 )
