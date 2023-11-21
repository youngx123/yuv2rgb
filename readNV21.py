# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 19:39 2023-11-15
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2


class NV21Reader:
    def __init__(self, size):
        self.height, self.width = size
        self.frame_len = int(self.width * self.height * 3 / 2)
        self.shape = (self.height, self.width)

    """
    def read(self, yuv_file):
        with open(yuv_file, "rb") as fid:
            data = fid.readlines()
            data = data[0]
        data = np.frombuffer(data, dtype=np.uint8)

        y = data[:self.width * self.height]
        # y = y[0:self.width*self.height:2]

        y1 = y.reshape(self.height, self.width)
        print(y1.shape)
        y1 = y1[0:-1:2, 0:-1:2]
        print(y1.shape)
        plt.imshow(y1)
        plt.show()

        uv = data[self.width * self.height:]
        uv_lenth = len(uv)
        v = uv[0:uv_lenth:2]
        u = uv[1:uv_lenth:2]

        # v = np.repeat(v, 2, axis=0)
        # u = np.repeat(u, 2, axis=0)

        y = y1.reshape((int(self.height // 2), self.width // 2, 1))
        u = u.reshape((int(self.height // 2), self.width // 2, 1))
        v = v.reshape((int(self.height // 2), self.width // 2, 1))

        r = y + 1.402 * (v - 128)
        g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128)
        b = y + 1.772 * (u - 128)

        rgb = np.concatenate((r, g, b), axis=-1)
        plt.imshow(rgb / 255)
        plt.show()
    """

    def readCmat(self, yuv_file):
        """
        convert yuv to rgb with opencv
        :param yuv_file:
        :return:
        """
        with open(yuv_file, "rb") as fid:
            data = fid.readlines()
            data = data[0]
        data = np.frombuffer(data, dtype=np.uint8)

        yuv = data.reshape(int(self.height * 1.5), self.width)
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV21)
        plt.imshow(rgb)
        plt.show()
        return rgb

    def ndArray(self, yuv_file):
        """
        convert yuv to rgb with numpy
        :param yuv_file:
        :return:
        """
        with open(yuv_file, "rb") as fid:
            data = fid.readlines()
            data = data[0]
        data = np.frombuffer(data, dtype='<' + str(self.frame_len) + 'B')
        # yuv0 = data.reshape(int(self.height * 1.5), self.width)

        yuv = data.reshape(int(self.height * 1.5), self.width)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)

        for i in range(self.height):
            for j in range(self.width):
                y = yuv[i, j]
                ii = i // 2
                if j % 2 == 0:
                    jj = j
                else:
                    jj = j - 1

                v = yuv[self.height + ii, jj]
                u = yuv[self.height + ii, jj + 1]

                rgb[i, j, 0] = y
                rgb[i, j, 1] = u
                rgb[i, j, 2] = v

        m = np.array([
            [1.000, 1.000, 1.000],
            [0.000, -0.394, 2.032],
            [1.140, -0.581, 0.000],
        ])

        rgb[:, :, 1:] -= 0.5 * 255.0
        rgb1 = np.dot(rgb, m)

        plt.imshow(rgb1 / 255.0)
        plt.show()

    def readCmatConvert(self, yuv_file):
        with open(yuv_file, "rb") as fid:
            data = fid.readlines()
            data = data[0]
        data = np.frombuffer(data, dtype=np.uint8)

        yuv = data.reshape(int(self.height * 1.5), self.width)

        # convert to rgb
        y = yuv[:self.height, :]
        vu = yuv[self.height:, :]

        v = vu[:, 0::2]
        u = vu[:, 1::2]

        v = np.repeat(v, 2, axis=1)
        v = np.repeat(v, 2, axis=0)

        u = np.repeat(u, 2, axis=1)
        u = np.repeat(u, 2, axis=0)

        y = np.reshape(y, (self.height, self.width, 1))
        u = np.reshape(u, (self.height, self.width, 1)) - 127.5
        v = np.reshape(v, (self.height, self.width, 1)) - 127.5

        rgb00 = np.concatenate((y, u, v), axis=-1)
        m = np.array([
            [1.000, 1.000, 1.000],
            [0.000, -0.394, 2.032],
            [1.140, -0.581, 0.000],
        ])
        rgb0 = np.dot(rgb00, m)
        plt.title("rgb0")
        plt.imshow(rgb0 / 255.0)
        plt.show()

        r = 1 * y + 1.140 * v
        g = 1 * y - 0.394 * u - 0.581 * v
        b = 1 * y + 2.032 * u

        rgb1 = np.concatenate((r, g, b), axis=-1)
        plt.title("rgb1")
        plt.imshow(rgb1 / 255.0)
        plt.show()

        return yuv


class NV21Convert(nn.Module):
    def __init__(self, image_size):
        super(NV21Convert, self).__init__()
        """
        :param image_size: [width, height]
        """
        self.shape = image_size
        self.width = image_size[1]
        self.height = image_size[0]

    def forward0(self, yuv):
        """
        half image size
        :param yuv:  8UC1
        :return:
        """
        y = yuv[..., 0:self.height:2, 0::2]

        vu = yuv[..., self.height:, :]
        h, w = self.height // 2, self.width // 2
        v = vu[0, 0, :, 0::2]
        v = v.reshape((h, w, 1)) - 127.5

        u = vu[0, 0, :, 1::2, ]
        u = u.reshape((h, w, 1)) - 127.5

        y = y.reshape((h, w, 1))

        r = y + 1.140 * v
        g = y - 0.394 * u - 0.581 * v
        b = y + 2.032 * u

        rgb = torch.cat((b, g, r), dim=-1)
        return rgb

    def forward(self, yuv):
        """
        full image size
        :param yuv:  8UC1
        :return:
        """
        y = yuv[..., :self.height, :]  # full image size

        vu = yuv[0, 0, self.height:, :]

        v = vu[..., 0::2] - 127.5
        uv_h, uv_w = v.shape
        v = v.reshape(uv_h, uv_w, 1)
        v = torch.cat((v, v), dim=2)
        v = v.reshape(uv_h, -1)

        v = v.reshape(uv_h, 1, -1)
        v = torch.cat((v, v), dim=1)
        v = v.reshape((self.height, self.width, 1))

        u = vu[..., 1::2] - 127.5
        uv_h, uv_w = u.shape
        u = u.reshape(uv_h, uv_w, 1)
        u = torch.cat((u, u), dim=2)
        u = u.reshape(uv_h, -1)

        u = u.reshape(uv_h, 1, -1)
        u = torch.cat((u, u), dim=1)
        u = u.reshape((self.height, self.width, 1))

        y = y.reshape((self.height, self.width, 1))

        r = y + 1.140 * v
        g = y - 0.394 * u - 0.581 * v
        b = y + 2.032 * u

        rgb = torch.cat((b, g, r), dim=-1)
        return rgb


def nv21_read():
    folders = r"E:\camera_open"
    image_list = os.listdir(folders)
    save_folder = r"E:\SA8155\ca_quant"
    size = (720, 1280)
    yuv_reader = NV21Reader(size)

    for file in image_list:
        file_name = os.path.join(folders, file)
        yuv_input = yuv_reader.readCmatConvert(file_name)

        input_tensor = torch.from_numpy(yuv_input)
        input_tensor = input_tensor[None, None, ...]

        net = NV21Convert(size)
        res = net(input_tensor)
        print(res.shape)

        plt.title("net")
        plt.imshow(res.numpy() / 255.0)
        plt.show()


def exprot_onnx():
    size = (720, 1280)
    model = NV21Convert(size)
    input_tensor = torch.randn(1, 1, int(size[0] * 1.5), size[1])
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
    nv21_read()
    # exprot_onnx()
