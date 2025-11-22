import read_dataset
import benchmark
from tqdm import tqdm

MODEL_DIR = "./model/onnx"
QUANTIZED_DIR = "./model/onnx_quantized"

def compare_levenshtein_distance():
    pairs = read_dataset.read_dataset()

    sample = pairs[:1]

    total_cer = 0.0
    total_t_fp32 = 0.0
    total_t_int8 = 0.0

    for pair in tqdm(sample, desc="Processing samples"):
        # img_path = pair["img_path"]
        # text = pair["text"]
        # print(f"img_path: {img_path}")
        # print(f"text: {text}")
        # print("================================")

        cer, t_fp32, t_int8 = benchmark.compare_levenshtein_distance(
            fp32_dir=MODEL_DIR,
            int8_dir=QUANTIZED_DIR,
            image_path=pair["img_path"]
        )

        total_cer += cer
        total_t_fp32 += t_fp32
        total_t_int8 += t_int8

        # is_accuracy_same = benchmark.compare_accuracy(
        #     fp32_dir=MODEL_DIR,
        #     int8_dir=QUANTIZED_DIR,
        #     image_path=pair["img_path"]
        # )


        # print(f"Is accuracy same: {is_accuracy_same}")
        
    print(f"Average CER: {total_cer / len(sample):.4f}")
    print(f"Average FP32 latency: {total_t_fp32 / len(sample):.4f}s")
    print(f"Average INT8 latency: {total_t_int8 / len(sample):.4f}s")
    print("================================")
    

def compare_accuracy():
    pairs = read_dataset.read_dataset()
    sample = pairs[:10]
    misses = 0

    for pair in tqdm(sample, desc="Processing samples"):

        equal = benchmark.compare_accuracy(
            fp32_dir=MODEL_DIR,
            int8_dir=QUANTIZED_DIR,
            image_path=pair["img_path"]
        )

        if (not equal):
            misses += 1       

    print(f"Misses: {misses}")
    print(f"Accuracy: {1 - misses / len(sample):.4f}")
    print("================================")

def default_accuracy():
    """
    未量化的模型与真实文本的准确率
    """
    pairs = read_dataset.read_dataset()
    sample = pairs[:100]
    misses = 0

    for pair in tqdm(sample, desc="Processing samples"):

        equal = benchmark.original_accuracy(
            fp32_dir=MODEL_DIR,
            image_path=pair["img_path"],
            text=pair["text"]
        )

        if (not equal):
            misses += 1       

    print(f"Misses: {misses}")
    print(f"Accuracy: {1 - misses / len(sample):.4f}")
    print("================================")
    # 跑100个数据，准确率为0

def default_distance():
    """
    未量化的模型与真实文本的距离
    """
    pairs = read_dataset.read_dataset()
    sample = pairs[:100]
    total_distance = 0.0

    for pair in tqdm(sample, desc="Processing samples"):

        distance = benchmark.original_distance(
            fp32_dir=MODEL_DIR,
            image_path=pair["img_path"],
            text=pair["text"]
        )

        total_distance += distance       

    print(f"Average Distance: {total_distance / len(sample):.4f}")
    print("================================")
    # 跑100个数据，距离为Average Distance: 0.1258

def quantized_distance():
    """
    量化后的模型与真实文本的距离
    """
    pairs = read_dataset.read_dataset()
    sample = pairs[:100]
    total_distance = 0.0

    for pair in tqdm(sample, desc="Processing samples"):

        distance = benchmark.original_distance(
            fp32_dir=QUANTIZED_DIR,
            image_path=pair["img_path"],
            text=pair["text"]
        )

        total_distance += distance       

    print(f"Average Distance: {total_distance / len(sample):.4f}")
    print("================================")
    # 跑100个数据，距离为Average Distance: 0.1407

def compare_mse():
    """
    对比均方误差（MSE）：
    """
    pairs = read_dataset.read_dataset()
    sample = pairs[:100]
    total_mse_fp32_gt = 0.0
    total_mse_int8_gt = 0.0
    total_mse_fp32_int8 = 0.0

    for pair in tqdm(sample, desc="Processing samples"):
        mse_fp32_gt, mse_int8_gt, mse_fp32_int8, _, _ = benchmark.compare_mse(
            fp32_dir=MODEL_DIR,
            int8_dir=QUANTIZED_DIR,
            image_path=pair["img_path"],
            text=pair["text"]
        )
        total_mse_fp32_gt += mse_fp32_gt
        total_mse_int8_gt += mse_int8_gt
        total_mse_fp32_int8 += mse_fp32_int8

        # print(f"MSE(FP32 vs GT): {mse_fp32_gt:.4f}")
        # print(f"MSE(INT8 vs GT): {mse_int8_gt:.4f}")
        # print(f"MSE(FP32 vs INT8): {mse_fp32_int8:.4f}")
        # print("================================")

    print(f"Average MSE(FP32 vs GT): {total_mse_fp32_gt / len(sample):.4f}")
    print(f"Average MSE(INT8 vs GT): {total_mse_int8_gt / len(sample):.4f}")
    print(f"Average MSE(FP32 vs INT8): {total_mse_fp32_int8 / len(sample):.4f}")
    print("================================")
    # 100条数据：
    # Average MSE(FP32 vs GT): 0.7154
    # Average MSE(INT8 vs GT): 0.7322
    # Average MSE(FP32 vs INT8): 0.4114

if __name__ == "__main__":
    # compare_levenshtein_distance()
    # compare_accuracy()
    # default_accuracy()
    # default_distance()
    # quantized_distance()
    compare_mse()

    

    