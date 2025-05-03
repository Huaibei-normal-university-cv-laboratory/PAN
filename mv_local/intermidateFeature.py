import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.utils import make_grid

from components.Ours import OmniSR

if __name__ == '__main__':

    network = OmniSR(1, 1, 64, up_scale=4, res_num=4, block_num=1, bias=True,
                     window_size=16, pe=True, ffn_bias=True)
    model_path = 'epoch442_OmniSR.pth'
    model_spec = torch.load(model_path, map_location=torch.device("cpu"))

    own_state = network.state_dict()
    # print(own_state.keys())
    for name, param in model_spec.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                if name.find('tail') == -1:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))

    ref = cv2.imread('Brats18_2013_10_1_t1_26.png', cv2.IMREAD_UNCHANGED)
    lr = cv2.imread('Brats18_2013_10_1_t2_26.png', cv2.IMREAD_UNCHANGED)
    # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    ref = torch.tensor(ref).unsqueeze(0).float() / 255.
    # hr_t2 = torch.tensor(hr_t2).unsqueeze(0).float() / 255.
    lr = torch.tensor(lr).unsqueeze(0).float() / 255.
    lr = lr.unsqueeze(0)
    ref = ref.unsqueeze(0)
    # x = torch.rand((1, 1, 64, 64))
    # y = torch.rand((1, 1, 256, 256))
    z = network(lr, ref)
    print(z.shape)
    from components.Ours import z_maps, y1_maps, local_maps,x1_maps, MPB_maps, featureFusion_maps

    # print(features_maps)
    for idx, features in enumerate(featureFusion_maps):
        features = features.squeeze(0)
        aggregate = torch.mean(features, dim=0)
        plt.imshow(aggregate.detach().cpu().numpy(), cmap='gray')  # 使用 detach() 分离张量
        plt.axis('off')
        plt.savefig(f'featureFusion{idx}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    for idx, features in enumerate(MPB_maps):
        features = features.squeeze(0)
        aggregate = torch.mean(features, dim=0)
        plt.imshow(aggregate.detach().cpu().numpy(), cmap='gray')  # 使用 detach() 分离张量
        plt.axis('off')
        plt.savefig(f'MPB{idx}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    for idx, features in enumerate(y1_maps):
        features = features.squeeze(0)
        aggregate = torch.mean(features, dim=0)
        plt.imshow(aggregate.detach().cpu().numpy(), cmap='gray')  # 使用 detach() 分离张量
        plt.axis('off')
        plt.savefig(f'y{idx}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    for idx, features in enumerate(x1_maps):
        features = features.squeeze(0)
        aggregate = torch.mean(features, dim=0)
        plt.imshow(aggregate.detach().cpu().numpy(), cmap='gray')  # 使用 detach() 分离张量
        plt.axis('off')
        plt.savefig(f'x{idx}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    for idx, features in enumerate(local_maps):
        features = features.squeeze(0)
        aggregate = torch.mean(features, dim=0)
        plt.imshow(aggregate.detach().cpu().numpy(), cmap='gray')  # 使用 detach() 分离张量
        plt.axis('off')
        plt.savefig(f'local{idx}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    for idx, features in enumerate(z_maps):
        features = features.squeeze(0)
        aggregate = torch.mean(features, dim=0)
        plt.imshow(aggregate.detach().cpu().numpy(), cmap='gray')  # 使用 detach() 分离张量
        plt.axis('off')
        plt.savefig(f'z{idx}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
