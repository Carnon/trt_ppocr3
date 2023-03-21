import numpy as np
import cv2
import time
from nets.common import resize_norm_img, CTCLabelDecode


def test_trt():
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt

    TRT_LOGGER = trt.Logger()

    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    def get_engine(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    engine_path = "weights/ch_ptocr_v3_rec_infer.engine"
    img = cv2.imread("imgs/word_3.jpg")
    h, w = img.shape[:2]
    input_img = resize_norm_img(img, max_wh_ratio=w / h)
    input_img = np.expand_dims(input_img, axis=0)
    image_size = input_img.shape
    result_size = (1, int(image_size[-1] / 8), 6625)
    print(input_img.shape)
    print(result_size)
    engine = get_engine(engine_path)
    context = engine.create_execution_context()
    stream = cuda.Stream()

    context.set_binding_shape(0, image_size)
    input_size = trt.volume(image_size)
    output_size = trt.volume(result_size)
    #
    input_type = trt.nptype(engine.get_binding_dtype("data"))
    output_type = trt.nptype(engine.get_binding_dtype("prob"))
    #
    input_host_memory = cuda.pagelocked_empty(input_size, input_type)
    output_host_memory = cuda.pagelocked_empty(output_size, output_type)
    #
    input_cuda_memory = cuda.mem_alloc(input_host_memory.nbytes)
    output_cuda_memory = cuda.mem_alloc(output_host_memory.nbytes)
    #
    bindings = [int(input_cuda_memory), int(output_cuda_memory)]
    #
    inputs = HostDeviceMem(input_host_memory, input_cuda_memory)
    outputs = HostDeviceMem(output_host_memory, output_cuda_memory)
    inputs.host = input_img
    #

    start = time.time()
    cuda.memcpy_htod_async(inputs.device, inputs.host, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs.host, outputs.device, stream)
    stream.synchronize()
    end = time.time()

    print("cost time: ", end - start)
    result = outputs.host
    result = result.reshape(result_size)

    postprocess = CTCLabelDecode(character_dict_path="weights/ppocr_keys_v1.txt", use_space_char=False)
    result = postprocess(result)
    print(result)


def test_onnx():
    import onnxruntime as rt
    sess = rt.InferenceSession("weights/ch_infer.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    img = cv2.imread("6.png")
    h, w = img.shape[:2]
    input_img = resize_norm_img(img, max_wh_ratio=w / h)
    input_img = np.expand_dims(input_img, axis=0)
    start = time.time()
    result = sess.run([output_name], {input_name: input_img})[0]
    print(time.time() - start)
    postprocess = CTCLabelDecode(character_dict_path="weights/ppocr_keys_v1.txt", use_space_char=False)
    result = postprocess(result)
    print(result)


if __name__ == '__main__':
    test_trt()
    # test_onnx()
    print("ok")
