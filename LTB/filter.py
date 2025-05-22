import argparse
import json
import os
import pickle
import shutil
import zipfile
from collections import OrderedDict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import *
from shapely.geometry import Polygon
from tqdm import tqdm

from mmocr.core.evaluation.utils import boundary_iou


class Tool:
    def make_dir(self, path, clear_all=False):
        if not os.path.exists(path):
            os.makedirs(path)
        elif clear_all is True:
            shutil.rmtree(path)
            os.makedirs(path)
        msg = f":: Make a directory: '{path}'"
        return msg

    def zip_files(self, files_path_lst, zip_file_path):
        self.make_dir(os.path.dirname(zip_file_path))
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for file_path in files_path_lst:
                zipf.write(file_path, arcname=os.path.basename(file_path))
        msg = f"==> save a zip file at 'P{zip_file_path}'."
        return msg

    def json2dict(self, json_path):
        msg = f"==> loading json file '{json_path}' ..."
        try:
            with open(json_path, encoding='UTF-8') as f:  # , errors='ignore'
                data = json.load(f)
        except Exception as e:
            print(f"json file: {os.path.abspath(json_path)}")
            raise e
        # print("::> loaded.")
        return data

    def write_json_file(self, content, file_path):
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False, indent=4)
        with open(file_path, 'w', encoding='UTF-8') as outfile:
            outfile.write(content)
        msg = f"==> save a file to '{file_path}'."
        return msg

    def write_lines(self, lines_lst, file_path):
        with open(file_path, 'w', encoding='UTF-8') as f:
            f.writelines(lines_lst)
        msg = f"--> save a file to '{file_path}'."
        return msg

    def points2coco_bbox(self, point_lst):
        point = [float(_) for _ in point_lst]
        point = np.reshape(point, (-1, 2))
        x_min, y_min = np.min(point, axis=0)
        x_max, y_max = np.max(point, axis=0)
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def coco_bbox2points(self, bbox):
        x, y, w, h = bbox
        return [x, y, x + w, y, x + w, y + h, x, y + h]

    def points2rect(self, point_lst):
        point = [float(_) for _ in point_lst]
        point = np.reshape(point, (-1, 2))
        x_min, y_min = np.min(point, axis=0)
        x_max, y_max = np.max(point, axis=0)
        return [x_min, y_min, x_max, y_max]

    def points2vert(self, points_lst, target='list'):
        assert len(points_lst) % 2 == 0
        if target == 'tuple':
            vertices = [(points_lst[i], points_lst[i + 1]) for i in range(0, len(points_lst), 2)]
        elif target == 'list':
            vertices = [[points_lst[i], points_lst[i + 1]] for i in range(0, len(points_lst), 2)]
        return vertices

    def draw_line(self, draw, segm, fill='red', width=3, order=True):
        count = len(segm) // 2
        if order:
            for i in range(count):
                draw.text((segm[i * 2] + 1, segm[i * 2 + 1] + 1), str(i + 1), fill=fill)
        segm.append(segm[0])
        segm.append(segm[1])
        for i in range(count):
            x0, y0, x1, y1 = segm[i * 2: i * 2 + 4]
            draw.line([x0, y0, x1, y1], fill=fill, width=width)
        return draw

    def random_color(self):
        x = int(np.random.choice(range(256)))
        y = int(np.random.choice(range(256)))
        z = int(np.random.choice(range(256)))
        rgb_color = tuple((x, y, z))

        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(*rgb)

        hex_color = rgb_to_hex(rgb_color)
        return hex_color
        # return "#{}{}{}{}{}{}".format(*(random.choice("0123456789abcdef") for _ in range(6)))
        # np.random.randint(0, 255, size=(3, ))

    def check_polys(self, points_lst):
        vertices = [(points_lst[j], points_lst[j + 1]) for j in range(0, len(points_lst), 2)]
        try:
            pdet = Polygon(vertices)
            assert pdet.is_valid
        except Exception as e:
            print(e)
            return False
        pRing = LinearRing(vertices)
        if not pRing.is_ccw:
            vertices.reverse()
        points_lst = [_ for coord in vertices for _ in coord]
        return points_lst

    def make_points_unique(self, points_lst):
        vertices = np.array(points_lst).reshape(-1, 2).tolist()
        unique_vertices_dict = OrderedDict()
        for vertex in vertices:
            unique_vertices_dict[tuple(vertex)] = vertex
        unique_vertices = list(unique_vertices_dict.values())
        if np.array(points_lst).ndim == 1:
            unique_target = np.array(unique_vertices).reshape(-1).tolist()
        else:
            assert np.array(points_lst).ndim == 2
            unique_target = np.array(unique_vertices).reshape(-1, 2).tolist()
        return unique_target

    def make_points_ccw(self, points_lst, ccw=True):
        vertices = [(points_lst[j], points_lst[j + 1]) for j in range(0, len(points_lst), 2)]

        pdet = Polygon(vertices)
        if not pdet.is_valid:
            pdet = pdet.buffer(0)
        assert pdet.is_valid
        try:
            vertices = list(pdet.exterior.coords)
        except Exception as e:
            print(f"make points ccw failed: {e}\n"
                  f"{points_lst}\n")
            raise e
        pRing = LinearRing(vertices)
        if pRing.is_ccw != ccw:
            vertices.reverse()
        unique_vertices = self.make_points_unique(vertices)
        new_points_lst = [_ for coord in unique_vertices for _ in coord]
        return new_points_lst

    def generate_adet_format_based_on_coco_anno(self, adet_file, saved_dir, zip_file):
        print(f"==> converting adet annotation '{adet_file}' into lexicons ...")
        coco_data = self.json2dict(adet_file)
        id2name, gts = dict(), dict()
        for image in coco_data["images"]:
            id = image["id"]
            fname, _ = os.path.splitext(os.path.normpath(image["file_name"]))
            id2name[str(id)] = fname
            gts[fname] = list()
        progress = tqdm(coco_data["annotations"])
        for anno in progress:
            bbox = anno["bbox"]
            points = self.coco_bbox2points(bbox)
            image_id = anno["image_id"]
            fname, _ = os.path.splitext(id2name[str(image_id)])
            polys = self.make_points_ccw(points, False)
            polys = [str(_) for _ in polys]
            line = ",".join(polys)
            line += ",####\n"
            gts[fname].append(line)
        self.make_dir(saved_dir, True)
        files_lst = list()
        for fname, lines in gts.items():
            fpath = os.path.join(saved_dir, fname + ".txt")
            files_lst.append(fpath)
            self.write_lines(lines, fpath)
        self.zip_files(files_lst, zip_file)

        print(f"==> Lexicons are saved to '{saved_dir}' and '{zip_file}'.")

    def compare_preds_and_gts(self, img_path, preds, gts_ignored, gts_cared):
        # if False:  # 没有pickle文件
        #     tool.make_dir(args.vis_dir, True)
        #     model = init_detector(args.config, args.checkpoint, device=args.device)
        #     if hasattr(model, 'module'):
        #         model = model.module
        #     test_data = tool.json2dict(args.ann_file)
        #     progress_bar = tqdm(test_data['images'])
        #
        #     img2ann_idx = dict()
        #     for ann_idx, ann in enumerate(test_data['annotations']):
        #         image_id = str(ann['image_id'])
        #         if image_id not in img2ann_idx:
        #             img2ann_idx[image_id] = [ann_idx]
        #         else:
        #             img2ann_idx[image_id].append(ann_idx)
        #
        #     for image in progress_bar:
        #         image_path = os.path.join(args.img_root, image['file_name'])
        #         preds = model_inference(model, image_path)['boundary_result']
        #         gts_crowd = [test_data['annotations'][ann_idx]['segmentation'][0] for ann_idx in img2ann_idx[str(image['id'])] if test_data['annotations'][ann_idx]['iscrowd']]
        #         gts_not_crowd = [test_data['annotations'][ann_idx]['segmentation'][0] for ann_idx in img2ann_idx[str(image['id'])] if not test_data['annotations'][ann_idx]['iscrowd']]
        #
        #         pil_img = Image.open(image_path).convert("RGBA")
        #         img_draw = ImageDraw.Draw(pil_img)
        #
        #         for gt in gts_crowd:
        #             img_draw.polygon(gt, outline='blue', width=5)
        #         for gt in gts_not_crowd:
        #             img_draw.polygon(gt, outline='red', width=5)
        #
        #         overlay = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
        #         over_draw = ImageDraw.Draw(overlay)
        #         for pred in preds:
        #             points, score = pred[:-1], pred[-1]
        #             if score < 0.5:
        #                 continue
        #             over_draw.polygon(points, fill=(255, 255, 0, int(255 * 0.2)))
        #
        #         save_path = os.path.join(args.vis_dir, os.path.basename(image['file_name'])[:-3] + 'png')
        #         combined = Image.alpha_composite(pil_img, overlay)
        #         combined.save(save_path)
        pil_img = Image.open(img_path).convert("RGBA")
        img_draw = ImageDraw.Draw(pil_img)

        for gt in gts_ignored:
            img_draw.polygon(gt, outline='blue', width=1)
        for gt in gts_cared:
            img_draw.polygon(gt, outline='red', width=3)
            # img_draw.polygon(gt, fill=(255, 0, 0, int(255 * 0.3)))  # red
        overlay = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
        over_draw = ImageDraw.Draw(overlay)
        for pred in preds:
            points, score = pred[:-1], pred[-1]
            if score < 0.5:
                continue
            over_draw.polygon(points, fill=(255, 255, 0, int(255 * 0.3)))
            # over_draw.polygon(points, outline='yellow', width=3)
        combined = Image.alpha_composite(pil_img, overlay)
        return combined

    def load_pickle_file(self, pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def write_pickle_file(self, content, file_path):
        self.make_dir(os.path.dirname(file_path))
        with open(file_path, 'wb') as file:
            pickle.dump(content, file)
        msg = f"==> save a file to '{file_path}'."
        return msg

    def poly2min_rect(self, poly):
        rect = cv2.minAreaRect(np.array(poly).astype(int).reshape(-1, 2))
        box = cv2.boxPoints(rect)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        return box.tolist()

    def generate_img2anns(self, coco_data, fname2idx=False, image_id2image_idx=False):
        images, annotations = coco_data['images'], coco_data['annotations']
        img_id2anns = dict()
        img_name2img_idx = dict()
        img_id2img_idx = dict()
        for img_idx, image in enumerate(images):
            img_name2img_idx[image['file_name']] = img_idx
            img_id2img_idx[str(image['id'])] = img_idx
            img_id = str(image['id'])
            if not img_id in img_id2anns:
                img_id2anns[str(img_id)] = list()
        for ann_idx, ann in enumerate(annotations):
            img_id = str(ann['image_id'])
            img_id2anns[img_id].append(ann_idx)

        if fname2idx:
            return img_id2anns, img_name2img_idx

        if image_id2image_idx:
            return img_id2anns, img_id2img_idx

        return img_id2anns

    def calc_iou_given_mask(self, mask1, mask2):
        """
        Args:
            mask1: torch.tensor(np.array()), dtype=torch.float32)
            mask2: torch.tensor(np.array()), dtype=torch.float32)
        Returns: Intersection over Union
        """
        intersection = mask1 * mask2
        union = torch.clamp(mask1 + mask2, 0, 1)
        intersection_area = torch.sum(intersection)
        union_area = torch.sum(union)
        iou = intersection_area / union_area
        return iou

    def poly2binary_mask(self, poly, width, height):
        mask = np.zeros((width, height), np.uint8)
        polygon_vertices = np.array(poly, dtype=np.int32).reshape(-1, 2)
        contours = [polygon_vertices]
        cv2.drawContours(mask, contours, -1, (1), -1)
        return mask

    def calc_iou_given_poly(self, poly1, poly2):
        rect1 = np.array(self.points2rect(poly1), dtype=np.int32)
        rect2 = np.array(self.points2rect(poly2), dtype=np.int32)
        x_min, x_max = min(rect1[0], rect2[0]), max(rect1[2], rect2[2])
        y_min, y_max = min(rect1[1], rect2[3]), max(rect1[1], rect2[3])
        transformed_poly1, transformed_poly2 = np.array(poly1, np.int32), np.array(poly2, np.int32)
        for _ in range(0, len(poly1), 2):
            transformed_poly1[_] -= x_min
            transformed_poly1[_ + 1] -= y_min
        for _ in range(0, len(poly2), 2):
            transformed_poly2[_] -= x_min
            transformed_poly2[_ + 1] -= y_min
        # print(x_min, x_max, y_min, y_max)
        # print(transformed_poly1, transformed_poly2, sep='\n')
        mask1 = self.poly2binary_mask(transformed_poly1, x_max, y_max)
        mask2 = self.poly2binary_mask(transformed_poly2, x_max, y_max)
        mask1 = torch.tensor(np.array(mask1), dtype=torch.float32)
        mask2 = torch.tensor(np.array(mask2), dtype=torch.float32)
        iou = self.calc_iou_given_mask(mask1, mask2)
        return iou

    def vis_img_based_on_coco(self, images_dir, vis_dir, coco_path, suspend=True):
        anno_data = self.json2dict(coco_path)
        self.make_dir(vis_dir, True)
        image_progress = tqdm(anno_data["images"][:])
        for image in image_progress:
            image_id = image['id']
            img_pth = os.path.join(images_dir, image['file_name'])
            pil_img = Image.open(img_pth)

            draw = ImageDraw.Draw(pil_img)
            save_pth = os.path.join(vis_dir, os.path.basename(img_pth)[:-4] + "_vis.jpg")
            for anno in anno_data["annotations"]:
                if anno['image_id'] != image_id:
                    continue
                if "polys" in anno:
                    self.draw_line(draw, anno["polys"], order=False)
                elif 'segmentation' in anno:
                    color = 'blue' if anno['iscrowd'] else 'red'
                    self.draw_line(draw, anno["segmentation"][0], fill=color, order=False)
            pil_img.save(save_pth)
            if suspend:
                input(f"{img_pth} --> {save_pth}\ncontinue？")


tool = Tool()


def parse_args():
    parser = argparse.ArgumentParser(description="Screen for text instances that fail detectors.")

    parser.add_argument('--annotation-file', type=str, required=True,
                        help='Path to the JSON file in COCO instances annotation format.')
    parser.add_argument('--img-prefix', type=str, required=True,
                        help='The directory where the images in the JSON file exist.')
    parser.add_argument('--det-results', type=str, nargs='+', required=True,
                        help='List of paths to pickle files, each for the prediction result of a certain detector.')
    parser.add_argument('--vis-dir', type=str, default=None,
                        help='Directory to save visualization images of failed text instances. '
                             'If provided, visualizations will be created.')
    parser.add_argument('--iou-thr', type=float, default=0.5,
                        help='IoU threshold for the screening process.')

    args = parser.parse_args()
    return args


def screen_for_failed_text_instances(
        annotation_file,
        img_prefix,
        det_results,
        vis_dir=None,
        iou_thr=0.5):
    """
    Screen for the text instances that fail detectors
    through a comparison between the detection results and GT annotation.

    Args:
        annotation_file (str): path to the json file in coco instances annotation format.
        img_prefix (str): the directory where the images in this json file exists.
        det_results (list[str]): list of paths to pickle files, each for the prediction result of a certain detector
            Content of pickle files should have the following format:
                [ // List of dictionaries, one for each image.
                    {
                        'boundary_result': [ // List of bounding lists, one for each text instance
                            [x1, y1, x2, y2, ..., score], // Bounding polygons + confidence score
                            ...
                        ],
                        'filename': "the filename of corresponding image"
                    },
                    ...
                ]
        vis_dir (str or none): the directory where to save visualization images of failed text instances.
            When provided, this function will do the following:
                * Outline the text instances that fail the detectors with red bounding boxes.
                * Mask the detection polygon that intersect with the failed text instances in yellow.
                * Save the visualization images to the directory provided.
        iou_thr (float): iou threshold for the screening process.
            If all bounding boxes predicted by the detectors have an Intersection over Union (IoU) value
            with a given text instance that is below a certain threshold,
            then this text instance is considered to fail all detectors.

    Returns:
        failed_images (list): list of filename of images that contains any text instance that fails all detectors.
            List content is in coco annotation format.
        failed_text_instances (list): list of text instances that fail all detectors.
            List content is in coco annotation format.
    """

    gt_data = tool.json2dict(annotation_file)
    images, annotations = gt_data['images'], gt_data['annotations']
    img_idx2ann_idx = tool.generate_img2anns(gt_data)

    # merge all prediction results
    img_name2preds = {os.path.basename(_['file_name']): list() for _ in images}
    for pkl_file in det_results:
        pkl_data = tool.load_pickle_file(pkl_file)
        for pred_data in pkl_data:
            _key = os.path.basename(pred_data['filename'])
            _val = pred_data['boundary_result']
            img_name2preds[_key].extend(_val)

    tool.make_dir(vis_dir, True)
    failed_images, failed_text_instances = list(), list()
    image_progress = tqdm(images)
    ignored_num, cared_num, detected_num = 0, 0, 0

    for image in image_progress:
        image_path = os.path.join(img_prefix, image['file_name'])
        assert os.path.exists(image_path)
        image_name = os.path.basename(image_path)
        preds = img_name2preds[image_name]
        try:
            gts_ignored_idx = [ann_idx for ann_idx in img_idx2ann_idx[str(image['id'])] if
                               annotations[ann_idx]['iscrowd']]
            gts_cared_idx = [ann_idx for ann_idx in img_idx2ann_idx[str(image['id'])] if
                             not annotations[ann_idx]['iscrowd']]
            # gts_ignored_instances = [annotations[ann_idx]['segmentation'][0] for ann_idx in
            #              img_idx2ann_idx[str(image['id'])] if
            #              annotations[ann_idx]['iscrowd']]
            # gts_cared_instances = [annotations[ann_idx]['segmentation'][0] for ann_idx in
            #                  img_idx2ann_idx[str(image['id'])] if
            #                  not annotations[ann_idx]['iscrowd']]
        except Exception as KeyError:
            # KeyError: there is no text instance in the image according to the gt annotation
            continue

        ignored_num += len(gts_ignored_idx)
        cared_num += len(gts_cared_idx)

        vis_gts_cared_instances, vis_preds_failed_instances = list(), list()
        for gt_idx in gts_cared_idx:
            if len(preds) != 0:
                annotation = annotations[gt_idx]
                gt = annotation['segmentation'][0]
                iou_lst = list()
                for pred in preds:
                    iou = boundary_iou(gt, pred[:-1])
                    iou_lst.append(iou)
                max_iou = max(iou_lst)
                if max_iou > iou_thr:
                    detected_num += 1
                    continue
            else:
                # print(f"Image '{image_name}': no text instances were detected.")
                continue

            vis_gts_cared_instances.append(gt)
            for i, iou in enumerate(iou_lst):
                if 0 < iou < iou_thr:
                    vis_preds_failed_instances.append(preds[i])

        # there exist some text instances that fail all detectors
        if len(vis_gts_cared_instances):
            failed_images.append(image)
            failed_text_instances.append(annotation)

            if vis_dir:
                combined = tool.compare_preds_and_gts(image_path, vis_preds_failed_instances, [],
                                                      vis_gts_cared_instances)
                save_path = os.path.join(vis_dir, os.path.basename(image['file_name'])[:-3] + 'png')
                combined.save(save_path)

    print(f"Ignored: {ignored_num}\tCared: {cared_num}\n"
          f"Detected: {detected_num}\tFailed: {cared_num - detected_num}")

    return failed_images, failed_text_instances


if __name__ == "__main__":
    """
    Example
    >> python filter.py \
    --annotation-file icdar2015/icdar2015_test.json \
    --img-prefix icdar2015/imgs/test \
    --det-results inference/preds/abcnet_v1_icdar2015.pkl inference/preds/dbpp_icdar2015.pkl \
    --vis-dir inference/ic15_dbpp_abcnet_pred_error \
    --iou-thr 0.5
    """

    args = parse_args()
    failed_images, failed_text_instances = screen_for_failed_text_instances(
        annotation_file=args.annotation_file,
        img_prefix=args.img_prefix,
        det_results=args.det_results,
        vis_dir=args.vis_dir,
        iou_thr=args.iou_thr
    )