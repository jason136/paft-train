# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
from typing import Optional
import argparse
import onnxruntime_extensions


def get_yolo_model(version: int, onnx_model_name: str):
    # install yolov8
    from pip._internal import main as pipmain
    try:
        import ultralytics
    except ImportError:
        pipmain(['install', 'ultralytics'])
        import ultralytics
    pt_model = Path(f"yolov{version}n.pt")
    model = ultralytics.YOLO(str(pt_model))  # load a pretrained model
    exported_filename = model.export(format="onnx")  # export the model to ONNX format
    assert exported_filename, f"Failed to export yolov{version}n.pt to onnx"
    import shutil
    destination_path = Path(onnx_model_name)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(exported_filename).resolve() != destination_path.resolve():
        shutil.move(exported_filename, destination_path)


def add_pre_post_processing_to_yolo(input_model_file: Path, output_model_file: Path):
    """Construct the pipeline for an end2end model with pre and post processing. 
    The final model can take raw image binary as inputs and output the result in raw image file.

    Args:
        input_model_file (Path): The onnx yolo model.
        output_model_file (Path): where to save the final onnx model.
    """
    from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
    # Match the exported YOLO model opset (typically 22) to avoid mismatch errors
    add_ppp.yolo_detection(input_model_file, output_model_file, "jpg", onnx_opset=22)


def run_inference(onnx_model_file: Path, test_image_file: Optional[Path] = None):
    import onnxruntime as ort
    import numpy as np

    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())

    image_path = test_image_file if test_image_file is not None else Path("19041780_d6fd803de0_3k.jpg")
    image = np.frombuffer(open(image_path, 'rb').read(), dtype=np.uint8)
    session = ort.InferenceSession(str(onnx_model_file), providers=providers, sess_options=session_options)

    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: image}
    output = session.run(['image_out'], inp)[0]
    output_filename = Path('yolo_result.jpg')
    open(output_filename, 'wb').write(output)
    try:
        from PIL import Image
        Image.open(output_filename).show()
    except Exception:
        # Display is optional; ignore errors if Pillow/display is unavailable
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export YOLOvN to ONNX, add pre/post, and run inference.')
    parser.add_argument('--version', type=int, default=8, help='YOLO major version (e.g., 5 or 8).')
    parser.add_argument('--test_image', type=str, default=None, help='Path to a JPG test image.')
    args = parser.parse_args()

    version = args.version
    onnx_model_name = Path(f"yolov{version}n.onnx")
    if not onnx_model_name.exists():
        print("Fetching original model...")
        get_yolo_model(version, str(onnx_model_name))

    onnx_e2e_model_name = onnx_model_name.with_suffix(suffix=".with_pre_post_processing.onnx")
    print("Adding pre/post processing...")
    add_pre_post_processing_to_yolo(onnx_model_name, onnx_e2e_model_name)
    print("Testing updated model...")
    test_image_path = Path(args.test_image) if args.test_image else None
    run_inference(onnx_e2e_model_name, test_image_path)
