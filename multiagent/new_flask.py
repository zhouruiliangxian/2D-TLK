import os
import numpy as np
import cv2
import torch
from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import matplotlib.pyplot as plt
from copy import deepcopy

from flask import Flask, request, jsonify
import os
import cv2
import numpy as np

app = Flask(__name__)


def reparameterize_model(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    if not inplace:
        model = deepcopy(model)

    def _fuse(m):
        for child_name, child in m.named_children():
            if hasattr(child, "fuse"):
                setattr(m, child_name, child.fuse())
            elif hasattr(child, "reparameterize"):
                child.reparameterize()
            elif hasattr(child, "switch_to_deploy"):
                child.switch_to_deploy()
            _fuse(child)

    _fuse(model)
    return model


config_file = r"D:\deepLearning\mmsegmentation-main\lraspp_ket_fastvit_ful\lraspp_ket_fastvit_ful.py"
checkpoint_file = r"D:\deepLearning\mmsegmentation-main\lraspp_ket_fastvit_ful\best_mIoU_iter_16000.pth"
# device = 'cpu'
device = "cuda:0"
model = init_model(config_file, checkpoint_file, device=device)
model = reparameterize_model(model)
model.eval()


def main(img_bgr_file, save_path):
    img_bgr_file = os.path.normpath(img_bgr_file)

    # print('x')
    palette = [
        ["background", [127, 127, 127]],
        ["monolayer", [255, 255, 204]],
        ["bilayer", [204, 153, 51]],
        ["multilayer", [51, 102, 102]],
    ]
    palette_dict = {}
    for idx, each in enumerate(palette):
        palette_dict[idx] = each[1]

    img_bgr = cv2.imread(img_bgr_file)

    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
    print(pred_mask.shape)
    pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    for idx in palette_dict.keys():
        pred_mask_bgr[np.where(pred_mask == idx)] = palette_dict[idx]
    pred_mask_bgr = pred_mask_bgr.astype("uint8")

    # save_path = os.path.join(r"out\val_Re", "val-pred-" + img_bgr_file.split("/")[-1])

    cv2.imwrite(save_path, pred_mask_bgr)

    unique_values, counts = np.unique(pred_mask, return_counts=True)

    # 创建一个字典来存储类别名称、颜色值和数量
    result = {}
    palette = [
        ["background", [127, 127, 127]],
        ["monolayer", [204, 255, 255]],
        ["bilayer", [51, 153, 204]],
        ["multilayer", [102, 102, 51]],
    ]

    # 遍历 unique_values 和 counts
    for value, count in zip(unique_values, counts):
        if value < len(palette):  # 确保 value 在 palette 的索引范围内
            category_name = palette[value][0]  # 获取类别名称
            color = palette[value][1]  # 获取颜色值
            result[category_name] = {"color": color, "count": count}

    # 打印结果
    res = ""
    for category, info in result.items():
        res += (
            f"Class: {category}, Color_RGB: {info['color']}, Count: {info['count']} pixes"
            + "\n"
        )

    return res


# API端点，用于接收文件路径并返回处理后的结果
@app.route("/process_image", methods=["POST"])
def process_image():
    # 从请求中获取图片路径和保存路径
    img_bgr_file = request.form.get("img_bgr_file")
    save_path = request.form.get("save_path")

    if not img_bgr_file or not save_path:
        return jsonify({"error": "Both img_bgr_file and save_path are required"}), 400

    # 检查文件是否存在
    if not os.path.exists(img_bgr_file):
        return jsonify({"error": "Input image file does not exist"}), 400

    # 调用主函数处理图像
    result = main(img_bgr_file, save_path)

    return jsonify({"message": "Image processed successfully", "result": result})


if __name__ == "__main__":
    app.run(debug=True)
