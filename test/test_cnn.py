import onnxruntime as ort
import time
import numpy as np
import cv2
import argparse



CPU_BACKEND = "CPUExecutionProvider"
GPU_BACKEND = "DmlExecutionProvider"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackbird Results Thresholder")
    parser.add_argument(
        "model_path",
        metavar="<MODEL_PATH>",
        help="Path to model as a *.onnx file.",
    )
    args = parser.parse_args()


    print(ort.get_device())


    image_data = np.zeros([1,3,224,224],dtype=np.float32)


    """
    # Use this to change from HWC to CHW
    imin = cv2.imread("test/subimage.png")
    imin = cv2.cvtColor(imin, cv2.COLOR_BGR2RGB)
    # CNN pre-processing
    imin = imin.astype(np.float32)
    #subimg /= 255. # NOTE: Normalization layer inside the original CNNs!!!!!
    imin = np.expand_dims(imin, axis=0)
    image_data = np.transpose(imin, (0,3,1,2)) # NHWC to NCHW (ONNX)
    """

    sess = ort.InferenceSession(args.model_path, providers=[GPU_BACKEND])

    print(sess.get_provider_options())

    # get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
    input_name = sess.get_inputs()[0].name
    print("Input Name: ",input_name)

    # get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
    output_name = sess.get_outputs()[0].name
    print("Output Name: ",output_name)

    for _ in range(3):
        start = time.time()
        # inference run using image_data as the input to the model 
        out = sess.run([output_name], {input_name: image_data})[0]
        end = time.time()

        total_time = (end - start)* 1000
        print("Inference time:",total_time,"ms")

    print("Output shape:", out.shape)
    print(out)