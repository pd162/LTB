#### 📊 Evaluation

We provide 15 `json` files annotated in [MMOCR-1.x OCRDataset format](https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html#ocrdataset).

- `challenging_<categoty_name>.json`

  each for different challenge categories: blurred, artistic, glass, single-character, distorted, inverse, delimited, dense, overlapped, occluded, low-contrast, complex-background, and other.

- `challenging_norm.json`

  All text instances retain its *care/don't care* label as in the official data.

- `challenging_hard.json`

  Only text instances that belong to at least one challenge categories are labeled as *care*.

#### 🌪️ Filter

We provide filter.py to screen for the text instances missed by several detectors.

```bash
python filter.py \
--annotation-file /path/to/annotation.json \
--img-prefix /path/to/images \
--det-results /path/to/det1.pkl /path/to/det2.pkl \
--vis-dir /path/to/output/dir \
--iou-thr 0.5
```

Note:

- The content of the annotation file should be in [coco instances annotation format](https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html#ocrdataset).

- The content of the pickle file (det-results)  should have the following format:

  ```json
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
  ```