from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_DIR = "./model/onnx"
QUANTIZE_DIR = "./model/onnx_quantized"

encode_model_path = f'{MODEL_DIR}/encoder_model.onnx'
decode_model_path = f'{MODEL_DIR}/decoder_model.onnx'
decoder_model_merged_path = f'{MODEL_DIR}/decoder_model_merged.onnx'
decoder_with_past_model_path = f'{MODEL_DIR}/decoder_with_past_model.onnx'

encode_quant_model_path = f'{QUANTIZE_DIR}/encoder_model.quant.onnx'
decode_quant_model_path = f'{QUANTIZE_DIR}/decoder_model.quant.onnx'
decoder_model_merged_quant_path = f'{QUANTIZE_DIR}/decoder_model_merged.quant.onnx'
decoder_with_past_model_quant_path = f'{QUANTIZE_DIR}/decoder_with_past_model.quant.onnx'

def quantize_model():
    # 运行测试
    # run_test()
    quantize_dynamic(decode_model_path, decode_quant_model_path, weight_type=QuantType.QUInt8)
    quantize_dynamic(encode_model_path, encode_quant_model_path, weight_type=QuantType.QUInt8)
    quantize_dynamic(decoder_model_merged_path, decoder_model_merged_quant_path, weight_type=QuantType.QUInt8)
    quantize_dynamic(decoder_with_past_model_path, decoder_with_past_model_quant_path, weight_type=QuantType.QUInt8)

    print('quantized model saved to', decode_quant_model_path, encode_quant_model_path)

if __name__ == '__main__':
    quantize_model()