# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 9:34 2023-11-16
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2


class UYVYReader:
    def __init__(self, size):
        self.height, self.width = size
        self.frame_len = int(self.width * self.height * 2)
        self.shape = (self.height, self.width)

    def readCmat(self, yuv_file):
        with open(yuv_file, "rb") as fid:
            data = fid.readlines()
            data = data[0]
        data = np.frombuffer(data, dtype=np.uint8)

        yuv = data.reshape(self.height, self.width, 2)

        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_UYVY)
        plt.title("opencv to rgb")
        plt.imshow(rgb)
        plt.show()
        return yuv

    def readCmatConvert(self, yuv_file):
        with open(yuv_file, "rb") as fid:
            data = fid.readlines()
            data = data[0]
        data = np.frombuffer(data, dtype=np.uint8)

        yuv = data.reshape((self.height, self.width, 2))

        uv = yuv[..., 0]
        y = yuv[..., 1]

        strid = 1
        dst_h, dst_w = self.height // strid, self.width // strid
        u = uv[..., 0:self.width:2]
        u = u.repeat(2, 1)
        v = uv[..., 1:self.width:2]
        v = v.repeat(2, 1)

        y = y[0:self.height:strid, 0:self.width:strid]

        y = y.reshape((dst_h, dst_w, 1))
        u = u.reshape((self.height, self.width, 1)) - 127.5
        v = v.reshape((self.height, self.width, 1)) - 127.5

        r = 1 * y + 0 * u + 1.140 * v
        g = 1 * y - 0.394 * u - 0.581 * v
        b = 1 * y + 2.032 * u + 0 * v

        rgb = np.concatenate((r, g, b), axis=-1)
        rgb = rgb.clip(0, 255)
        plt.title("CmatConvert ")
        plt.imshow(rgb / 255)
        plt.show()
        return yuv


class UYVYConvert(nn.Module):
    def __init__(self, image_size):
        super(UYVYConvert, self).__init__()
        """
        :param image_size: [width, height]
        """
        self.shape = image_size
        self.width = image_size[1]
        self.height = image_size[0]

    def forward0(self, yuv):
        """
        转为除图像外的任意偶数尺寸
        :param yuv:  8UC2
        :return:
        """
        strid = 4
        uv = yuv[0, 0, ...]
        y = yuv[0, 1, 0::strid, 0::strid]

        h, w = self.height // strid, self.width // strid
        u = uv[0::strid, 0::strid]
        u = u.reshape((h, w, 1)) - 127.5

        v = uv[0::strid, 1::strid]
        v = v.reshape((h, w, 1)) - 127.5

        y = y.reshape((h, w, 1))

        r = 1 * y + 0 * u + 1.140 * v
        g = 1 * y - 0.394 * u - 0.581 * v
        b = 1 * y + 2.032 * u + 0 * v

        rgb = torch.cat((r, g, b), dim=-1)
        return rgb

    def forward(self, yuv):
        """
        转为原始图像尺寸
        :param yuv:  8UC2
        :return:
        """
        uv = yuv[0, 0, ...]
        y = yuv[0, 1, ...]

        u = uv[:, 0::2]
        u = u.reshape(self.height, -1, 1)
        u = torch.cat((u, u), dim=2)
        u = u.reshape(self.height, -1)
        u = u.reshape((self.height, self.width, 1)) - 127.5

        v = uv[:, 1::2]
        v = v.reshape(self.height, -1, 1)
        v = torch.cat((v, v), dim=2)
        v = v.reshape(self.height, -1)
        v = v.reshape((self.height, self.width, 1)) - 127.5

        y = y.reshape((self.height, self.width, 1))

        r = 1 * y + 0 * u + 1.140 * v
        g = 1 * y - 0.394 * u - 0.581 * v
        b = 1 * y + 2.032 * u + 0 * v

        rgb = torch.cat((r, g, b), dim=-1)
        return rgb


def uyvy_read():
    folders = r"D:\SOTA_model\uyvy_data\frame4"
    image_list = os.listdir(folders)
    save_folder = r"E:\SA8155\ca_quant"
    size = (1300, 1600)
    yuv_reader = UYVYReader(size)

    for file in image_list:
        file_name = os.path.join(folders, file)
        yuv_input = yuv_reader.readCmatConvert(file_name)
        # break
        yuv_input = np.transpose(yuv_input, (2, 0, 1))
        input_tensor = torch.from_numpy(yuv_input)
        input_tensor = input_tensor[None, ...]

        net = UYVYConvert(size)
        res = net(input_tensor)
        print(res.shape)

        plt.title("net")
        plt.imshow(res.numpy() / 255.0)
        plt.show()


def exprot_onnx():
    """
    export uyvy Convert model to onnx
    :return:
    """
    size = size = (1300, 1600)
    model = UYVYConvert(size)
    input_tensor = torch.randn(1, 2, size[0], size[1])
    onnx_name = "nv21_half_size.onnx"
    model.eval()

    torch.onnx.export(
        model,
        input_tensor,
        onnx_name,
        input_names=["image"],
        output_names=["output"],
        opset_version=11,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=1
    )


if __name__ == "__main__":
    """
    a = torch.from_numpy(np.array([[1,2,3,4,5],[6,7,8,9,0]]))
    a1 = a.reshape(2,-1, 1)
    a2 = torch.cat((a1,a1), dim=2)
    a3 = a2.reshape(2,-1)

    a4 = a3.reshape(2, 1, 10)
    a5 = torch.cat((a4, a4), dim=1)
    a5 = a5.reshape(4,-1)
    """

    uyvy_read()

    exprot_onnx()
