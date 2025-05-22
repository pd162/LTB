import numpy as np
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from paddleocr import PaddleOCR
import torch
import cv2

ocr = PaddleOCR(
    lang="en",
    use_gpu=True,
    det=True,
    use_angle_cls=True,
    rec=False
)


class TextDataset(ImageFolder):
    def __init__(
            self,
            root,
            transform=None
    ):
        super(TextDataset, self).__init__(root)
        self.root = root
        self.transform = transform

    def get_det_res(self, path):
        ocr_res = ocr.ocr(path)[0]
        det_res = []
        if ocr_res is not None:
            for res in ocr_res:
                det = np.array(res[0]).reshape(-1)
                det_res.append(det)
        return np.array(det_res)

    def get_aux_mask(self, sample, ocr_det_res):
        trans_h, trans_w = sample.shape[1], sample.shape[2]
        aux_mask = np.zeros([trans_h, trans_w], dtype=np.uint8)
        if len(ocr_det_res) > 0:
            for det in ocr_det_res:
                points = np.array(det, dtype=np.int32).reshape((-1, 1, 2))
                aux_mask = cv2.fillPoly(aux_mask, [points], 255)
        return aux_mask

    def __getitem__(self, index):
        # add text det mask
        path, target = self.samples[index]
        sample = self.loader(path)
        ori_w, ori_h = sample.size
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        trans_h, trans_w = sample.shape[1], sample.shape[2]
        try:
            ocr_det_res = self.get_det_res(path)
            ratio_h = 1. * trans_h / ori_h
            ratio_w = 1. * trans_w / ori_w
            ocr_det_res[..., ::2] = ocr_det_res[..., ::2] * ratio_w
            ocr_det_res[..., 1::2] = ocr_det_res[..., 1::2] * ratio_h
            aux_mask = self.get_aux_mask(sample, ocr_det_res)
        except:
            aux_mask = np.zeros([trans_h, trans_w], dtype=np.uint8)

        return sample, aux_mask


if __name__ == '__main__':
    root = '/data2/xxx/data/laion_ocr_lite'
    import torchvision.transforms as transforms
    transform_train = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = TextDataset(root, transform=transform_train)
    data_loader = torch.utils.data.DataLoader(dataset)
    for item in data_loader:
        print(item)