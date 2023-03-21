import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from nets.backbone import MobileNetV1Enhance
from nets.head import CTCHead
from nets.neck import SequenceEncoder


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.backbone = MobileNetV1Enhance(in_channels=3)
        self.neck = SequenceEncoder(self.backbone.out_channels)
        self.head = CTCHead(self.neck.out_channels)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


def resize_norm_img(img, max_wh_ratio):
    rec_image_shape = (3, 48, 480)
    limited_max_width = 1280
    limited_min_width = 16

    imgC, imgH, imgW = rec_image_shape

    assert imgC == img.shape[2]
    max_wh_ratio = max(max_wh_ratio, imgW / imgH)
    imgW = int((imgH * max_wh_ratio))
    imgW = max(min(imgW, limited_max_width), limited_min_width)
    h, w = img.shape[:2]
    ratio = w / float(h)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    if ratio_imgH > imgW:
        resized_w = imgW
    else:
        resized_w = int(ratio_imgH)
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path, use_space_char=False):

        self.beg_str = "sos"
        self.end_str = "eos"

        self.character_str = []
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path,
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char=True)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, torch.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)

        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


def test_main():
    base_model = BaseModel().cpu()
    ckt = torch.load("weights/ch_ptocr_v3_rec_infer.pth", map_location="cpu")
    base_model.load_state_dict(ckt)
    base_model.eval()

    # img = cv2.imread("word_3.jpg")
    img = cv2.imread("6.png")
    h, w = img.shape[:2]
    input_img = resize_norm_img(img, max_wh_ratio=w/h)
    with torch.no_grad():
        inp = torch.from_numpy(np.expand_dims(input_img, 0))
        print("input shape: ", inp.shape)
        prob_out = base_model(inp)
        print(prob_out)
        print("output shape: ", prob_out.shape)

    postprocess = CTCLabelDecode(character_dict_path="weights/ppocr_keys_v1.txt", use_space_char=False)
    result = postprocess(prob_out.cpu().numpy())
    print(result)
    pass


def test_export():
    base_model = BaseModel().cpu()
    ckt = torch.load("weights/ch_ptocr_v3_rec_infer.pth", map_location="cpu")
    base_model.load_state_dict(ckt)
    base_model.eval()

    x = torch.randn(size=(1, 3, 48, 480))
    torch.onnx.export(
        base_model,
        x,
        "weights/ch_infer.onnx",
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 3: "width"},
                      "output": {0: "batch_size", 1: "width"}}
    )


def test_onnx():
    import onnxruntime as ort
    sess = ort.InferenceSession("weights/ch_rec_infer.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    input_x = np.random.random(size=(1, 3, 48, 3200)).astype(np.float32)
    output_y = sess.run([output_name], {input_name: input_x})[0]
    output_y2 = sess.run([output_name], {input_name: input_x})[0]
    output_y3 = sess.run([output_name], {input_name: input_x})[0]
    print(output_y)
    print(output_y.shape)
    print(output_y2.shape)
    print(output_y3.shape)


if __name__ == '__main__':
    # test_main()
    test_export()
    # test_onnx()
    print("ok")

