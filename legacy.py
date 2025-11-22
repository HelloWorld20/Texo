# import onnx

import os
import sys
from PIL import Image
import numpy as np
from typing import Any
from numpy.typing import NDArray
import onnxruntime as ort
from onnxruntime import InferenceSession
# 本地运行：将项目的 src 目录加入 sys.path，支持导入 src/texo/*
PROJECT_ROOT = os.path.dirname(__file__)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from transformers import AutoTokenizer
from texo.data.processor import EvalMERImageProcessor, TextProcessor
from optimum.onnxruntime import ORTModelForVision2Seq


# image_path = './TechnoSelection/test_img/单行公式.png'
image_path = './TechnoSelection/test_img/多行公式.png'

MODEL_DIR = "./model/onnx"
MAX_NEW_TOKENS = 128

encode_model_path = f'{MODEL_DIR}/encoder_model.onnx'
decode_model_path = f'{MODEL_DIR}/decoder_model.onnx'

def preprocessImg():
    """
    使用与训练一致的图像处理器进行预处理，避免尺寸/归一化不匹配。
    返回符合 ONNX 编码器输入的 `pixel_values` 张量 (numpy.float32)。
    """
    image = Image.open(image_path).convert("RGB")
    processor = EvalMERImageProcessor(image_size={"height": 384, "width": 384})
    pixel_values = processor(image).numpy()  # [1, 3, 384, 384], float32
    return pixel_values

def encoder(pixel_value: NDArray[Any]):
    """
    调用编码器 ONNX，输出 encoder_hidden_states。
    同时打印输入/输出名称以便诊断接口是否匹配。
    """
    enc = InferenceSession(encode_model_path)
    print("[Encoder] inputs:", [i.name for i in enc.get_inputs()])
    print("[Encoder] outputs:", [o.name for o in enc.get_outputs()])
    ort_inputs = {"pixel_values": pixel_value}
    ort_outputs = enc.run(None, ort_inputs)
    encoder_hidden_states = ort_outputs[0]
    shape = getattr(encoder_hidden_states, "shape", None)
    print("encoder_hidden_states:", shape if shape is not None else type(encoder_hidden_states))
    return encoder_hidden_states

def decoder(encoder_hidden_states: NDArray[Any]):
    """
    创建解码器 ONNX Session，并做一次干运行以打印输出形状。
    注意：实际生成建议使用 ORTModelForVision2Seq.generate 以自动处理 KV cache。
    """
    dec = InferenceSession(decode_model_path)
    print("[Decoder] inputs:", [i.name for i in dec.get_inputs()])
    print("[Decoder] outputs:", [o.name for o in dec.get_outputs()])
    ort_inputs = {
        # 不同导出配置可能是 "decoder_input_ids" 或 "input_ids"，这里优先尝试 "input_ids"
        "input_ids": np.zeros((1, 1), dtype=np.int64),
        "encoder_hidden_states": encoder_hidden_states,
    }
    try:
        ort_outputs = dec.run(None, ort_inputs)
        out0 = ort_outputs[0]
        shape = getattr(out0, "shape", None)
        print("decoder_outputs:", shape if shape is not None else type(out0))
    except Exception as e:
        print("[Decoder] 试运行失败:", e)
    return dec

def choose_bos_eos(tokenizer: Any):
    """
    兼容性地找一个 BOS / EOS token id。
    不同模型的配置可能不一样，这里做几层 fallback。
    """
    bos_id = None
    eos_id = None

    # BOS
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(tokenizer, "cls_token_id", None)
    if bos_id is None:
        bos_id = getattr(tokenizer, "pad_token_id", None)
    if bos_id is None:
        bos_id = 0  # 兜底

    # EOS
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "sep_token_id", None)
    # 允许 None：没有就不强制停止

    return bos_id, eos_id

def greedy_generate(
    encoder_hidden_states: NDArray[Any],
    dec_sess: ort.InferenceSession,
    tokenizer: Any,
    max_new_tokens: int = 128,
) -> str:
    """
    用贪心解码（每步取 argmax）从 decoder ONNX 生成文本。
    这里用的是不带 KV cache 的简单版 decoder_model.onnx 接口：
      inputs:
        - input_ids: [batch, tgt_len] int64
        - encoder_hidden_states: [batch, src_len, hidden_size] float32
    """
    bos_id, eos_id = choose_bos_eos(tokenizer)
    print(f"[INFO] BOS id = {bos_id}, EOS id = {eos_id}")

    # 初始只喂一个 BOS token
    input_ids = np.array([[bos_id]], dtype=np.int64)

    for step in range(max_new_tokens):
        decoder_inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }

        outputs = dec_sess.run(None, decoder_inputs)
        # 一般第 0 个输出是 logits: [batch, tgt_len, vocab_size]
        logits = np.asarray(outputs[0])
        # 取最后一个位置的 logits
        last_logits = logits[:, -1, :]  # [batch, vocab_size]
        next_id = int(last_logits.argmax(axis=-1)[0])

        # 拼到序列后面
        input_ids = np.concatenate(
            [input_ids, np.array([[next_id]], dtype=np.int64)],
            axis=1,
        )

        if eos_id is not None and next_id == eos_id:
            print(f"[INFO] Hit EOS at step {step}")
            break

    # 通常会把 BOS/EOS 去掉
    generated_ids = input_ids[0].tolist()
    # 去掉第一个 bos_id
    if generated_ids and generated_ids[0] == bos_id:
        generated_ids = generated_ids[1:]

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text


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
    """
    主入口：
    - 首先走推荐的 ORT 生成路径，确保输出与训练/导出保持一致。
    - 如需对比手写贪心，可取消注释下方代码进行对照。
    """
    # 手写贪心（可选对照）
    pixel_value = preprocessImg()
    encoder_hidden_states = encoder(pixel_value)
    dec = decoder(encoder_hidden_states)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    text = greedy_generate(
        encoder_hidden_states=encoder_hidden_states,
        dec_sess=dec,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print("----------------")
    print(text)
    print("----------------")

# model_fp32 = 'path/to/the/model.onnx'
# model_quant = 'path/to/the/model.quant.onnx'
# quantized_model = quantize_dynamic(model_fp32, model_quant)

# python -m onnxruntime.quantization.preprocess --input /Users/leon.w/workspace/python/Texo/model/resnet50_Opset18/resnetv2_50_Opset18.onnx --output /Users/leon.w/workspace/python/Texo/model/resnet50_Opset18/resnetv2_50_Opset18-infer.onnx