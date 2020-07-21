import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader
    loader_train = get_loader(config.image_dir_train, config.attr_path_train, config.selected_attrs,
                                 config.chexpert_crop_size, config.image_size, config.batch_size,
                                 'RSNA_Pneumonia', config.mode, config.num_workers)

    loader_test = get_loader(config.image_dir_valid, config.attr_path_valid, config.selected_attrs,
                                 config.chexpert_crop_size, config.image_size, config.batch_size,
                                 'RSNA_Pneumonia', config.mode, config.num_workers)

    loader_simul = get_loader(config.image_dir_valid, config.attr_path_simul,
                                config.selected_attrs, config.chexpert_crop_size, config.image_size, config.batch_size,
                                'RSNA_Pneumonia', 'eval', config.num_workers)


    solver = Solver(loader_train, loader_test, loader_simul, config)
    solver.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=2, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--chexpert_crop_size', type=int, default=300, help = 'crop size for the CheXpert dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=1, help='weight for identity loss')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='RSNA_Pneumonia', choices=['RSNA_Pneumonia'])
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Normal', 'Abnormal'])

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Directories.
    parser.add_argument('--image_dir_train', type=str, default='./data/images')
    parser.add_argument('--image_dir_valid', type=str, default='./data/images')

    # Train/Valid /Simul datafile
    parser.add_argument('--attr_path_train', type=str, default='./data/rsna_pneumonia_train.txt')
    parser.add_argument('--attr_path_valid', type=str, default='./data/rsna_pneumonia_valid.txt')
    parser.add_argument('--attr_path_simul', type=str, default='./data/rsna_pneumonia_simul.txt')

    parser.add_argument('--model_save_dir', type=str, default='./model/save')
    parser.add_argument('--sample_dir', type=str, default='./model/samples')
    parser.add_argument('--ground_truth_file', type=str, default="./data/bounding_box.pickle")
    parser.add_argument('--ground_truth', type=str, default='./data/bounding_box.pickle')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=500)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)