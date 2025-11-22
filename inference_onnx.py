import os
from PIL import Image
import numpy as np

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

from transformers import AutoTokenizer
from texo.data.processor import EvalMERImageProcessor, TextProcessor
from optimum.onnxruntime import ORTModelForVision2Seq

# MODEL_DIR = "./model/onnx"
MODEL_DIR = './model/onnx_quantized'
# image_path = './TechnoSelection/test_img/单行公式.png'
image_path = './TechnoSelection/test_img/多行公式.png'

def _onnx_dir_ready(path: str) -> bool:
    """
    检查指定目录中是否存在 ONNX 权重文件（*.onnx）。
    若不存在，返回 False，用于在运行前给出明确提示。
    """
    try:
        files = os.listdir(path)
    except Exception:
        return False
    return any(name.endswith(".onnx") for name in files)

def run_ort_generate() -> None:
    """
    使用 Optimum 的 ORTModelForVision2Seq 直接进行生成，并通过 tokenizer 解码。
    该路径会自动处理注意力掩码与 KV cache，比手写贪心更可靠。
    """
    device = "cuda" if ort.get_device() == "GPU" else "cpu"
    print(f"[INFO] 使用设备: {device}")

    # 允许通过环境变量覆盖 ONNX 目录
    onnx_dir = os.environ.get("TEXO_ONNX_DIR", MODEL_DIR)
    try:
        _files = os.listdir(onnx_dir)
        print(f"[INFO] ONNX 目录: {onnx_dir}")
        print(f"[INFO] 目录文件: {_files}")
    except Exception as e:
        print(f"[ERROR] 无法读取目录 '{onnx_dir}': {e}")
        return
    if not _onnx_dir_ready(onnx_dir):
        print(f"[ERROR] 在 '{onnx_dir}' 下未检测到任何 .onnx 权重文件。")
        print("        请先导出 ONNX（scripts/python/export_onnx.py），或将 TEXO_ONNX_DIR 指向正确目录。")
        return

    # 加载 ONNX 模型与 tokenizer（使用项目内 tokenizer 以保证词表一致）
    onnx_model = ORTModelForVision2Seq.from_pretrained(onnx_dir)
    text_processor = TextProcessor(config={
        "tokenizer_path": "data/tokenizer",
        "tokenizer_config":{
            "add_special_tokens": True,
            "max_length": 1024,
            "padding": "longest",
            "truncation": True,
            "return_tensors": "pt",
            "return_attention_mask": False,
        }
    })

    # 图像预处理
    image = Image.open(image_path).convert("RGB")
    processor = EvalMERImageProcessor(image_size={"height": 384, "width": 384})
    pixel_values = processor(image).unsqueeze(0)  # torch.Tensor [1, 3, 384, 384]

    # 生成与解码
    with np.errstate(all="ignore"):
        outputs = onnx_model.generate(pixel_values)
    text = text_processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("================ 生成结果 ================")
    print(text)
    print("========================================")

if __name__ == "__main__":
    run_ort_generate()