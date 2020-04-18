import copy
import torch
# import dsntnn
import torchvision
import torch.nn as nn
import sys, os





class Swish(nn.Module):
    
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, x):
        '''
        Forward pass of the function.
        '''
        return x * torch.sigmoid(x) 



class ResBlock(nn.Module):
    def __init__(self, n_dim):
        super(ResBlock, self).__init__()

        self.n_dim = n_dim

        self.fc1 = nn.Linear(n_dim, n_dim)
        self.fc2 = nn.Linear(n_dim, n_dim)
        self.acfun = nn.LeakyReLU()

    def forward(self, x0):

        x = self.acfun(self.fc1(x0))
        x = self.acfun(self.fc2(x))
        x = x+x0
        return x



class BodyGlobalPoseVAE(nn.Module):
    def __init__(self,zdim,num_hidden=512,f_dim=32,test=False,in_dim=3,
                 pretrained_resnet='resnet18.pth'):
        super(BodyGlobalPoseVAE, self).__init__()


        self.test = test
        self.zdim = zdim

        resnet = torchvision.models.resnet18()
        if pretrained_resnet is not None:
            print('[INFO][%s] Using pretrained resnet18 weights.'.format(self.__class__.__name__))
            resnet.load_state_dict(torch.load(pretrained_resnet))
        removed = list(resnet.children())[1:6]
        self.resnet = nn.Sequential(nn.Conv2d(in_dim, 64, kernel_size=7, 
                                              stride=2, padding=3,bias=False),
                                    *removed)
        self.conv = nn.Conv2d(128,f_dim,3,1,1) # b x f_dim x 28 x 28
        self.fc = nn.Linear(f_dim*16*16,num_hidden)


        ############ encoder torso ############
        self.torso_linear = nn.Linear(3,num_hidden)
        
        ############ mix all conditions ############
        self.encode = nn.Sequential(ResBlock(n_dim=2*num_hidden),
                                    ResBlock(n_dim=2*num_hidden))


        self.mean_linear = nn.Linear(2*num_hidden,zdim)
        self.log_var_linear = nn.Linear(2*num_hidden,zdim)



        ############ decoder for the scene+torso feature #######
        self.decode = nn.Sequential(nn.Linear(num_hidden+zdim, f_dim),
                                    ResBlock(n_dim=f_dim),
                                    ResBlock(n_dim=f_dim),
                                    nn.Linear(f_dim, 3))


    def sampler(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = eps.cuda()
        return eps.mul(var).add_(mu)


    def forward(self,scene,torso=None):
        if self.test:
            b = scene.size(0)
            z = torch.Tensor(b,self.zdim).cuda()
            z.normal_(0,1)

            fscene_ = self.conv(self.resnet(scene))
            z_s = self.fc(fscene_.view(b,-1)) # bxnum_hidden 
            z_sg = torch.cat([z, z_s],dim=1)
            x_g_gen = self.decode(z_sg)
            
            return x_g_gen

        else:
            b = scene.size(0)

            ############# encoder ####################
            fscene_ = self.conv(self.resnet(scene))
            z_s = self.fc(fscene_.view(b,-1)) # bxnum_hidden

            ###### torso  ######
            ftorso = self.torso_linear(torso)

            fc_cls_torso = torch.cat((z_s,ftorso),dim=1)
            # fc_cls_torso = z_s+ftorso
            feature = self.encode(fc_cls_torso)

            mean = self.mean_linear(feature)
            log_var = self.log_var_linear(feature)


            ############# sampling #################
            z = self.sampler(mean, log_var)


            ############# decoder ##################
            z_sg = torch.cat([z, z_s],dim=1)
            x_g_rec = self.decode(z_sg)

            return x_g_rec, mean, log_var









class BodyLocalPoseVAE(nn.Module):
    def __init__(self,zdim,num_hidden=512,f_dim=128,test=False,in_dim=3,
                 pretrained_resnet='resnet18.pth'):
        super(BodyLocalPoseVAE, self).__init__()

        self.test = test
        self.zdim = zdim

        resnet = torchvision.models.resnet18()
        if pretrained_resnet is not None:
            print('[INFO][%s] Using pretrained resnet18 weights.'.format(self.__class__.__name__))
            resnet.load_state_dict(torch.load(pretrained_resnet))
        removed = list(resnet.children())[1:6]
        self.resnet = nn.Sequential(nn.Conv2d(in_dim, 64, kernel_size=7, 
                                              stride=2, padding=3,
                                              bias=False),
                                    *removed)
        self.conv = nn.Conv2d(128,f_dim,3,1,1) # b x f_dim x 28 x 28
        self.fc = nn.Linear(f_dim*16*16,num_hidden)

        

        ############ encoder torso & body branch ############
        self.torso_linear = nn.Linear(3,num_hidden)
        self.pose_linear = nn.Linear(72,num_hidden)

        ############ mix all conditions ############
        self.encode = nn.Sequential(ResBlock(n_dim=3*num_hidden),
                                    ResBlock(n_dim=3*num_hidden))


        self.mean_linear = nn.Linear(3*num_hidden,zdim)
        self.log_var_linear = nn.Linear(3*num_hidden,zdim)



        ############ decoder for the scene+torso feature #######
        self.decode = nn.Sequential(nn.Linear(2*num_hidden+zdim, f_dim),
                                    ResBlock(n_dim=f_dim),
                                    ResBlock(n_dim=f_dim),
                                    nn.Linear(f_dim, 72))



    def sampler(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = eps.cuda()
        return eps.mul(var).add_(mu)


    def forward(self,scene, torso=None, pose=None):
        if self.test:
            b = scene.size(0)
            z = torch.Tensor(b,self.zdim).cuda()
            z.normal_(0,1)

            fscene_ = self.conv(self.resnet(scene))
            z_s = self.fc(fscene_.view(b,-1)) # bxnum_hidden 
            z_g = self.torso_linear(torso)

            z_sgl = torch.cat([z, z_g, z_s],dim=1)
            x_g_gen = self.decode(z_sgl)

            return x_g_gen
        else:

            b = scene.size(0)

            ############# encoder ####################
            fscene_ = self.conv(self.resnet(scene))
            z_s = self.fc(fscene_.view(b,-1)) # bxnum_hidden

            z_g = self.torso_linear(torso)
            z_l = self.pose_linear(pose)

            z_sgl = torch.cat([z_l, z_g, z_s] ,dim=1)
            
            feature = self.encode(z_sgl)

            mean = self.mean_linear(feature)
            log_var = self.log_var_linear(feature)

            ############# sampling #################
            z = self.sampler(mean, log_var)

            ############# decoder ##################
            z_sgl = torch.cat([z, z_g, z_s],dim=1)
            x_l_rec = self.decode(z_sgl)

            return x_l_rec, mean, log_var












