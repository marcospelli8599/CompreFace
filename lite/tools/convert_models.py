"""Convierte los modelos de CompreFace a ONNX. Se ejecuta UNA vez, en un contenedor con TensorFlow 2.15 (ver convert.Dockerfile);
el servicio solo necesita ONNX Runtime.

  /models/facenet/20180402-114759.pb      FaceNet (VGGFace2), grafo congelado de TF1
  /models/age/checkpoint-14999.*          edad (Levi y Hassner, Inception v3), punto de control de TF1
  /compreface/embedding-calculator/...    pesos y redes de MTCNN, y la definición de la red de edad

Salida en /out: facenet.onnx, age.onnx, mtcnn_pnet.onnx, mtcnn_rnet.onnx, mtcnn_onet.onnx
"""
import sys

import numpy as np
import tensorflow as tf
import tf2onnx

OUT = "/out"
EC = "/compreface/embedding-calculator"
sys.path.insert(0, f"{EC}/srcext")
sys.path.insert(0, EC)

tf1 = tf.compat.v1
OPSET = 13


def convert_facenet():
    graph_def = tf1.GraphDef()
    with open("/models/facenet/20180402-114759.pb", "rb") as f:
        graph_def.ParseFromString(f.read())

    def fn(x):
        return tf1.import_graph_def(
            graph_def, input_map={"input:0": x, "phase_train:0": tf.constant(False)}, return_elements=["embeddings:0"], name=""
        )[0]

    wrapped = tf1.wrap_function(fn, [tf.TensorSpec([None, 160, 160, 3], tf.float32, name="input")])
    frozen = wrapped.graph.as_graph_def()
    tf2onnx.convert.from_graph_def(
        frozen,
        input_names=[wrapped.inputs[0].name],
        output_names=[wrapped.outputs[0].name],
        opset=OPSET,
        output_path=f"{OUT}/facenet.onnx",
    )
    print("facenet.onnx listo")


def convert_age():
    # helpers.py solo depende de numpy, TensorFlow y tf_slim: se carga por ruta para no arrastrar el resto de CompreFace.
    import importlib.util

    spec = importlib.util.spec_from_file_location("agegender_helpers", f"{EC}/src/services/facescan/plugins/agegender/helpers.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    graph = tf1.Graph()
    with graph.as_default():
        session = tf1.Session(graph=graph)
        images = tf1.placeholder(tf.float32, [None, 160, 160, 3], name="images")
        logits = helpers.inception_v3(8, images)
        probabilities = tf.nn.softmax(logits, name="probabilities")
        tf1.global_variables_initializer()
        tf1.train.Saver().restore(session, "/models/age/checkpoint-14999")
        frozen = tf1.graph_util.convert_variables_to_constants(session, graph.as_graph_def(), ["probabilities"])
    tf2onnx.convert.from_graph_def(
        frozen, input_names=["images:0"], output_names=["probabilities:0"], opset=OPSET, output_path=f"{OUT}/age.onnx"
    )
    print("age.onnx listo")


def convert_mtcnn():
    from mtcnn.network.factory import NetworkFactory

    pnet, rnet, onet = NetworkFactory().build_P_R_O_nets_from_file(f"{EC}/srcext/mtcnn/data/mtcnn_weights.npy")
    for name, model, shape in (
        ("pnet", pnet, (None, None, None, 3)),
        ("rnet", rnet, (None, 24, 24, 3)),
        ("onet", onet, (None, 48, 48, 3)),
    ):
        spec = (tf.TensorSpec(shape, tf.float32, name="input"),)
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=OPSET, output_path=f"{OUT}/mtcnn_{name}.onnx")
        print(f"mtcnn_{name}.onnx listo")


if __name__ == "__main__":
    convert_facenet()
    convert_age()
    convert_mtcnn()
