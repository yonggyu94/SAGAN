# Training script for tiny-imagenet.
# Again, this script has a lot of bugs everywhere.
import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import losses as L

from models.discriminators.snresnet64 import Omniglot_Discriminator, VGG_Discriminator, Animal_Discriminator
from models.generators.resnet64 import Omniglot_Generator, VGG_Generator2, Animal_Generator

from dataloader import omniglot_data_loader, vgg_data_loader, data_loader, animal_data_loader, celeba_data_loader
import utils


dev = 'cuda' if torch.cuda.is_available() else 'cpu'


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_version", type=str, default="3-shot")
    parser.add_argument('--dataset', type=str, default="vgg", help="omniglot or vgg or animal")
    parser.add_argument('--dataset_root', type=str, default="/home/nas1_userC/yonggyu/ECCV2020/dataset")
    parser.add_argument('--n_shot', type=str, default="3-shot")
    
    parser.add_argument('--resize_size', type=int, default=84)
    parser.add_argument('--crop_size', type=int, default=64)

    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate of Adam. default: 0.0002')
    parser.add_argument('--lr_decay_start', '-lds', type=int, default=10000,
                        help='Start point of learning rate decay. default: 50000')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--ckpt_root', type=str, default="")

    
    # Dataset configuration
    parser.add_argument('--cGAN', default=True, action='store_true',
                        help='to train cGAN, set this ``True``. default: False')
    parser.add_argument('--data_root', type=str, default='tiny-imagenet-200',
                        help='path to dataset root directory. default: tiny-imagenet-200')
    parser.add_argument('--batch_size', '-B', type=int, default=64,
                        help='mini-batch size of training data. default: 64')
    parser.add_argument('--eval_batch_size', '-eB', default=None,
                        help='mini-batch size of evaluation data. default: None')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for training data loader. default: 8')
    # Generator configuration
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=32,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=7,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    # Discriminator (Critic) configuration
    parser.add_argument('--dis_arch_concat', '-concat', default=False, action='store_true',
                        help='If use concat discriminator, set this true. default: False')
    parser.add_argument('--dis_emb', type=int, default=128,
                        help='Parameter for concat discriminator. default: 128')
    parser.add_argument('--dis_num_features', '-dnf', type=int, default=32,
                        help='Number of features of discriminator (a.k.a nplanes or ndf). default: 64')
    # Optimizer settings
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='beta1 (betas[0]) value of Adam. default: 0.0')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 (betas[1]) value of Adam. default: 0.9')
    # Training setting
    parser.add_argument('--seed', type=int, default=46,
                        help='Random seed. default: 46 (derived from Nogizaka46)')
    parser.add_argument('--max_iteration', '-N', type=int, default=100000,
                        help='Max iteration number of training. default: 100000')
    parser.add_argument('--n_dis', type=int, default=5,
                        help='Number of discriminator updater per generator updater. default: 5')
    parser.add_argument('--num_classes', '-nc', type=int, default=0,
                        help='Number of classes in training data. No need to set. default: 0')
    parser.add_argument('--loss_type', type=str, default='hinge',
                        help='loss function name. hinge (default) or dcgan.')
    parser.add_argument('--relativistic_loss', '-relloss', default=False, action='store_true',
                        help='Apply relativistic loss or not. default: False')
    parser.add_argument('--calc_FID', default=False, action='store_true',
                        help='If calculate FID score, set this ``True``. default: False')
    # Log and Save interval configuration
    parser.add_argument('--results_root', type=str, default='results',
                        help='Path to results directory. default: results')
    parser.add_argument('--no_tensorboard', action='store_true', default=False,
                        help='If you dislike tensorboard, set this ``False``. default: True')
    parser.add_argument('--no_image', action='store_true', default=False,
                        help='If you dislike saving images on tensorboard, set this ``True``. default: False')
    parser.add_argument('--checkpoint_interval', '-ci', type=int, default=100,
                        help='Interval of saving checkpoints (model and optimizer). default: 1000')
    parser.add_argument('--log_interval', '-li', type=int, default=100,
                        help='Interval of showing losses. default: 100')
    parser.add_argument('--eval_interval', '-ei', type=int, default=100,
                        help='Interval for evaluation (save images and FID calculation). default: 1000')
    parser.add_argument('--n_eval_batches', '-neb', type=int, default=100,
                        help='Number of mini-batches used in evaluation. default: 100')
    parser.add_argument('--n_fid_images', '-nfi', type=int, default=50,
                        help='Number of images to calculate FID. default: 5000')
    parser.add_argument('--test', default=False, action='store_true',
                        help='If test this python program, set this ``True``. default: False')
    # Resume training
    parser.add_argument('--args_path', default=None, help='Checkpoint args json path. default: None')
    parser.add_argument('--gen_ckpt_path', '-gcp', default=None,
                        help='Generator and optimizer checkpoint path. default: None')
    parser.add_argument('--dis_ckpt_path', '-dcp', default=None,
                        help='Discriminator and optimizer checkpoint path. default: None')
    args = parser.parse_args()
    return args


def sample_from_data(args, device, data_loader):
    real, y = next(data_loader)
    real, y = real.to(device), y.to(device)
    if not args.cGAN:
        y = None
    return real, y


def sample_from_gen(args, device, num_classes, gen):
    z = utils.sample_z(
        args.batch_size, args.gen_dim_z, device, args.gen_distribution
    )
    if args.cGAN:
        pseudo_y = utils.sample_pseudo_labels(
            num_classes, args.batch_size, device
        )
    else:
        pseudo_y = None

    fake = gen(z, pseudo_y)
    
    return fake, pseudo_y, z


def pick_fixed_img(args, train_loader, img_num):
    img_list = []
    label_list = []

    for i in range(4):
        x_data, y_data = sample_from_data(args, dev, train_loader)
        for j in range(x_data.size(0)):
            img_list.append(x_data[j])
            label_list.append(y_data[j])

    img_list = img_list[0: img_num]
    label_list = label_list[0: img_num]

    return img_list, label_list


def directory_path(args):
    output = "output"
    weight_path = os.path.join(output, args.exp_version, 'weight')
    img_path = os.path.join(output, args.exp_version, 'img')

    if os.path.exists(weight_path) is False:
        os.makedirs(weight_path)
    if os.path.exists(img_path) is False:
        os.makedirs(img_path)

    return weight_path, img_path


def data_loader2(args):
    if args.crop_size == 28:
        root_path = args.dataset_root
        data_root = os.path.join(root_path, args.dataset + "_%s" % args.n_shot, args.n_shot)
        print(data_root)
        train_loader, s_dlen, num_classes = omniglot_data_loader(
            root=data_root,
            batch_size=args.batch_size,
            resize_size=args.resize_size,
            crop_size=args.crop_size)
        print("omniglot data_loader")
    elif args.crop_size == 64:
        root_path = args.dataset_root
        data_root = os.path.join(root_path, args.dataset + "_%s" % args.n_shot, args.n_shot)
        print(data_root)
        train_loader, s_dlen, num_classes = vgg_data_loader(
            root=data_root,
            batch_size=args.batch_size,
            resize_size=args.resize_size,
            crop_size=args.crop_size)
        print("vgg data_loader, animal, cub")
    elif args.crop_size == 128:
        root_path = args.dataset_root
        data_root = os.path.join(root_path, args.dataset + "_%s" % args.n_shot, args.n_shot)
        train_loader, s_dlen, num_classes = animal_data_loader(
            root=data_root,
            batch_size=args.batch_size,
            resize_size=args.resize_size,
            crop_size=args.crop_size)
        print("animal data_loader")
    else:
        raise Exception("Enter omniglot or vgg or animal")

    train_loader = iter(utils.cycle(train_loader))
    return train_loader, s_dlen, num_classes


def select_model(args, _n_cls):
    print("selecting model")
    if args.dataset == "omniglot":
        gen = Omniglot_Generator(
            args.gen_num_features, args.gen_dim_z, bottom_width=7, activation=F.relu,
            num_classes=_n_cls, distribution=args.gen_distribution).to(dev)
        dis = Omniglot_Discriminator(args.dis_num_features, _n_cls, F.relu).to(dev)
    elif args.dataset == "vgg" or args.crop_size == 64:
        gen = VGG_Generator2(
            args.gen_num_features * 2, args.gen_dim_z, bottom_width=4, activation=F.relu,
            num_classes=_n_cls, distribution=args.gen_distribution).to(dev)
        dis = VGG_Discriminator(args.gen_num_features * 2, _n_cls, F.relu).to(dev)
        print(111)
    elif args.dataset == "animal":
        gen = Animal_Generator(
            args.gen_num_features * 2, args.gen_dim_z, bottom_width=4, activation=F.relu,
            num_classes=_n_cls, distribution=args.gen_distribution).to(dev)
        dis = Animal_Discriminator(args.gen_num_features * 2, _n_cls, F.relu).to(dev)
    else:
        raise Exception("Enter model omniglot or vgg or animal")
    
    start = 0
    
    if args.load:
        print("load model..")
        ckpt_root = args.ckpt_root
        
        gen_ckpt = os.path.join(ckpt_root, "gen_latest.pth.tar")
        dis_ckpt = os.path.join(ckpt_root, "dis_latest.pth.tar")
        
        gen_ckpt = torch.load(gen_ckpt)
        dis_ckpt = torch.load(dis_ckpt)
        
        gen.load_state_dict(gen_ckpt['model'])
        dis.load_state_dict(dis_ckpt['model'])
        
        gen.train()
        dis.train()
        
        start = (len(os.listdir(ckpt_root)) // 2 - 2) * 1000
        print("complete loading model at start number : %d" % start)
        
    return gen, dis, start


def main():
    args = get_args()
    weight_path, img_path = directory_path(args)

    # CUDA setting
    if not torch.cuda.is_available():
        raise ValueError("Should buy GPU!")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

    # dataloading
    train_loader, s_dlen, _n_cls = data_loader2(args)

    fixed_z = torch.randn(200, 10, 128)
    fixed_img_list, fixed_label_list = pick_fixed_img(args, train_loader, 200)

    # initialize model
    gen, dis, start = select_model(args, _n_cls)
    
    gen, dis = gen.cuda(), dis.cuda()
    
    opt_gen = optim.Adam(gen.parameters(), args.lr, (args.beta1, args.beta2))
    opt_dis = optim.Adam(dis.parameters(), args.lr, (args.beta1, args.beta2))

    gen_criterion = L.GenLoss(args.loss_type, args.relativistic_loss)
    dis_criterion = L.DisLoss(args.loss_type, args.relativistic_loss)

    # Training loop
    for n_iter in tqdm.tqdm(range(start, args.max_iteration)):

        if n_iter >= args.lr_decay_start:
            decay_lr(opt_gen, args.max_iteration, args.lr_decay_start, args.lr)
            decay_lr(opt_dis, args.max_iteration, args.lr_decay_start, args.lr)

        # ==================== Beginning of 1 iteration. ====================
        _l_g = .0
        cumulative_loss_dis = .0
        for i in range(args.n_dis):
            if i == 0:
                fake, pseudo_y, _ = sample_from_gen(args, dev, _n_cls, gen)
                dis_fake = dis(fake, pseudo_y).cuda()
                dis_real = None

                loss_gen = gen_criterion(dis_fake, dis_real)
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
                _l_g += loss_gen.item()

            fake, pseudo_y, _ = sample_from_gen(args, dev, _n_cls, gen)
            real, y = sample_from_data(args, dev, train_loader)

            dis_fake, dis_real = dis(fake, pseudo_y), dis(real, y)
            loss_dis = dis_criterion(dis_fake, dis_real)

            dis.zero_grad()
            loss_dis.backward()
            opt_dis.step()

            cumulative_loss_dis += loss_dis.item()
        # ==================== End of 1 iteration. ====================

        if n_iter % args.log_interval == 0:
            tqdm.tqdm.write(
                'iteration: {:07d}/{:07d}, loss gen: {:05f}, loss dis {:05f}'.format(
                    n_iter, args.max_iteration, _l_g, cumulative_loss_dis))

        if n_iter % args.checkpoint_interval == 0:
            #Save checkpoints!
            utils.save_checkpoints(args, n_iter, gen, opt_gen, dis, opt_dis, weight_path)
            utils.save_img(fixed_img_list, fixed_label_list, fixed_z, gen,
                           32, 28, img_path, n_iter, device=dev)
    if args.test:
        shutil.rmtree(args.results_root)

if __name__ == '__main__':
    main()
