from model import Generator
from model import Discriminator
from torch.autograd import Variable
import torch.nn.functional as F
import os
import time
import datetime
import cv2
from util import *
from PIL import Image
from PIL import ImageDraw
import pickle

def normalize(tensor):
    return (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))

def Gen_normal_target(c_org):
    batchsz, _ = c_org.shape
    c_trg = c_org.clone()
    for i in range(batchsz):
        c_trg[i,0] = 1
        c_trg[i,1] = 0
    return c_trg

def Gen_disease_target(c_org):
    batchsz, _ = c_org.shape
    c_trg = c_org.clone()
    for i in range(batchsz):
        c_trg[i, 0] = 0
        c_trg[i, 1] = 1
    return c_trg

def get_rect(x, y, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def draw_box(img, Bounding_box_list):
    if Bounding_box_list != []:
        # np_image = img
        img = Image.fromarray(img)
        for bbox in Bounding_box_list:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            draw = ImageDraw.Draw(img)
            rect = get_rect(x=x, y=y, width=w, height=h, angle=0.0)
            draw.polygon([tuple(p) for p in rect])
        return np.asarray(img)
    else:
        return img

class Solver(object):
    def __init__(self, loader_train, loader_test, loader_simul, config):
        """Initialize configurations."""
        self.config = config
        # Data loader.
        self.fixed_loader = loader_simul
        self.train_loader = loader_train
        self.test_loader = loader_test
        self.ground_truth = config.ground_truth

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step


        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)



        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels_origin(self, c_org, c_dim=2):
        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.clone()
            c_trg[:, i] = 1  # Reverse attribute value.
            c_trg[:, (i+1)%2] = 0
            c_trg_list.append(c_trg.to(self.device))

        return c_trg_list


    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.

        data_loader = self.train_loader

        with open(self.ground_truth, 'rb') as f:
            BBox_map = pickle.load(f)


        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(0, self.num_iters):
            self.G.train()
            self.D.train()
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            CheXpert_iter = iter(data_loader)
            _, Train_image, Train_c_org = next(CheXpert_iter)

            rand_idx = torch.randperm(Train_c_org.size(0))
            Train_c_trg = Train_c_org[rand_idx]

            Train_image_ = Train_image.to(self.device)  # Input images
            Train_c_org_ = Train_c_org.to(self.device)  # Original domain lables
            Train_c_trg_ = Train_c_trg.to(self.device)  # Target domain labels

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(Train_image_)
            d_loss_real = - torch.mean(out_src)

            d_loss_cls = self.classification_loss(out_cls, Train_c_org_, self.dataset)
            # Compute loss with fake images.
            x_fake = self.G(Train_image_, Train_c_trg_)

            # out_src : Fake or Real
            # out_cls : Classes information
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(Train_image_.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * Train_image_.data + (1 - alpha) * x_fake.data).requires_grad_(True)

            out_src, _ = self.D(x_hat)

            # What is the meaning of gradient penalty
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()


            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(Train_image_, Train_c_trg_)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)

                g_loss_cls = self.classification_loss(out_cls, Train_c_trg_, self.dataset)

                # Reconstruction loss
                x_reconst = self.G(x_fake, Train_c_org_)
                g_loss_rec = torch.mean(torch.abs(Train_image_ - x_reconst))

                # Identity loss
                g_loss_diff = torch.mean(torch.abs(Train_image_ - x_fake))

                g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_rec * g_loss_rec + self.lambda_id * g_loss_diff

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_diff'] = g_loss_diff.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)


            # generate sample synthesis image in virtual file
            if (i + 1) % self.sample_step == 0:

                data_iter = iter(self.fixed_loader)

                log = "Iteration [{}/{}]".format(i + 1, self.num_iters)
                print(log)
                horison_final=[]
                for m in range(0,self.batch_size):

                    file_namess, x_fixed, c_org = next(data_iter)
                    x_fixed = x_fixed.to(self.device)
                    c_fixed_list = self.create_labels_origin(c_org, self.c_dim)

                    with torch.no_grad():
                        x_fixed_list = []
                        for j in range(self.batch_size):
                            try:
                                Bounding_box_list = BBox_map[file_namess[j] + ".jpg"]
                            except:
                                Bounding_box_list = []

                            # Draw box
                            origin_img = to_image(x_fixed[j], c=1)
                            origin_img = origin_img.reshape((256, 256))
                            # Mass
                            try:
                                origin_img = draw_box(origin_img, [Bounding_box_list])
                            except:
                                origin_img = draw_box(origin_img, Bounding_box_list)

                            # Others
                            origin_img = origin_img.reshape((256, 256, 1))
                            origin_img = to_image_np(origin_img)

                            x_fixed_list.append(origin_img)
                        concat_origin = cv2.vconcat(x_fixed_list)
                        concat_other_attirbute = []

                        for c_fixed in c_fixed_list:
                            x_filters, x_fakeimg = self.G.Generate_TargetImage(x_fixed, c_fixed)

                            paired_img = []
                            for j in range(self.batch_size):
                                target_image = to_image((x_fakeimg[j]))
                                lesion_map = denorm(x_filters[j].cpu()).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                                lesion_map = cv2.applyColorMap(lesion_map, cv2.COLORMAP_JET)

                                concat_pair = cv2.hconcat([lesion_map, target_image])
                                paired_img.append(concat_pair)
                            vconcat_paired = cv2.vconcat(paired_img)
                            concat_other_attirbute.append(vconcat_paired)
                        hconcat_paired = cv2.hconcat(concat_other_attirbute)
                        compiled_image = cv2.hconcat([concat_origin, hconcat_paired])
                        horison_final.append(compiled_image)

                final_image = cv2.hconcat(horison_final)
                sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                im = Image.fromarray(final_image)
                im.save(sample_path)
                print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
                # save model