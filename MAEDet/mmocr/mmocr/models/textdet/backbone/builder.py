import mmocr.models.textdet as textdet

def build_backbone(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    backbone = textdet.backbone.__dict__[cfg.type](**param)

    return backbone