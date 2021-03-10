"""
@author: Kohei Watanabe
@contact: koheitech001[at]gmail.com
"""

import numpy as np


def draw_bbox_each(img, bbox, text, textcolor, bbcolor):

    from PIL import Image, ImageFilter, ImageDraw
    x1, y1 = bbox[0], bbox[1]
    #y, x = int(bbox[1] - bbox[3] / 2), int(bbox[0] - bbox[2] / 2)
    w, h = bbox[2], bbox[3]
    pilImg = Image.fromarray(np.uint8(img))
    draw = ImageDraw.Draw(pilImg)
    text_w, text_h = draw.textsize(text)
    label_y = y1 if y1 <= text_h else y1 - text_h
    draw.rectangle((x1, label_y, x1 + w, label_y + h), outline=bbcolor)
    draw.rectangle((x1, label_y, x1 + text_w, label_y + text_h), outline=bbcolor, fill=bbcolor)
    draw.text((x1, label_y), text, fill=textcolor)

    ret = np.asarray(pilImg)
    #ret.flags.writeable = True
    return ret

def draw_bbox(img, bbox, categories, textcolor, bbcolor):

    ret = np.copy(img)
    for b_p in bbox:
        if b_p["score"] < 0.55:
            continue
        
        score_round = round(b_p["score"], 2)
        cat = categories[b_p["category_id"]]
        print(b_p, cat)
        cat_score = cat + " : " + str(score_round)
        
        ret = draw_bbox_each(ret, b_p["bbox"], cat_score, textcolor, bbcolor)

    return ret

