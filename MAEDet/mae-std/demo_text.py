import os

import models_textmae
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2
import tqdm

imagenet_mean = np.array([0.5, 0.5, 0.5])
imagenet_std = np.array([0.5, 0.5, 0.5])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch8'):
    # build model
    model = getattr(models_textmae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    return x[0], im_masked[0], im_paste[0]
    # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]
    #
    # plt.subplot(1, 4, 1)
    # show_image(x[0], "original")
    #
    # plt.subplot(1, 4, 2)
    # show_image(im_masked[0], "masked")
    #
    # plt.subplot(1, 4, 3)
    # show_image(y[0], "reconstruction")
    #
    # plt.subplot(1, 4, 4)
    # show_image(im_paste[0], "reconstruction + visible")
    #
    # plt.show()


if __name__ == '__main__':
    img_dir = "/data/xxx/DB/datasets/total_text/train_images/"
    ckpt_path = "/data/xxx/mae-std/output_dir/checkpoint-200.pth"
    model_mae = prepare_model(ckpt_path, 'mae_vit_large_patch8')
    torch.manual_seed(2)
    print('Model loaded.')
    for path in tqdm.tqdm(os.listdir(img_dir)):
        if 'jpg' in path:
            img_path = os.path.join(img_dir, path)
            img = Image.open(img_path)
            img = img.resize((640, 640))
            img = np.array(img) / 255.
            img = img - imagenet_mean
            img = img / imagenet_std
            show_image(torch.tensor(img))
            print('MAE with pixel reconstruction:')
            img, mask_img, recon_img = run_one_image(img, model_mae)
            print(0)
            cv2.imwrite(f'img/{path}', (255 * img.cpu().numpy()).astype('int32'))
            cv2.imwrite(f'maskimg/{path}',  (255 * mask_img.cpu().numpy()).astype('int32'))
            cv2.imwrite(f'recoimg/{path}',  (255 * recon_img.cpu().numpy()).astype('int32'))

