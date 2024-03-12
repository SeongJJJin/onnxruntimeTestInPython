import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize, QuantType, QuantFormat, CalibrationMethod, quantize_static

# 반양자화 :: float32 -> float16
# model = onnx.load("model/bestFloat32.onnx")
# model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
# onnx.save(model_fp16, "model/bestFloat16IOFix.onnx")


# 정적 양자화 :: float32 -> float8
# model_fp32 = 'model/bestFloat32.onnx'
# quantized_model = quantize_dynamic(model_fp32, 'model/bestFloat8.onnx', weight_type=QuantType.QUInt8) # 가중치 > 0 ~ 255
