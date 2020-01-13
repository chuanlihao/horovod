import h5py
import io
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.compiler.tensorrt import trt_convert as trt

LOCAL_CHECKPOINT_FILE = 'checkpoint.h5'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
K.set_session(tf.Session(config=config))

# Load checkpoint.
with open(LOCAL_CHECKPOINT_FILE, 'rb') as f:
    best_model_bytes = f.read()

max_sales = 41551.0

def exp_rmspe(y_true, y_pred):
    """Competition evaluation metric, expects logarithic inputs."""
    pct = tf.square((tf.exp(y_true) - tf.exp(y_pred)) / tf.exp(y_true))
    # Compute mean excluding stores with zero denominator.
    x = tf.reduce_sum(tf.where(y_true > 0.001, pct, tf.zeros_like(pct)))
    y = tf.reduce_sum(tf.where(y_true > 0.001, tf.ones_like(pct), tf.zeros_like(pct)))
    return tf.sqrt(x / y)


def act_sigmoid_scaled(x):
    """Sigmoid scaled to logarithm of maximum sales scaled by 20%."""
    return tf.nn.sigmoid(x) * tf.log(max_sales) * 1.2

CUSTOM_OBJECTS = {'exp_rmspe': exp_rmspe,
                  'act_sigmoid_scaled': act_sigmoid_scaled}

def deserialize_model(model_bytes, load_model_fn):
    """Deserialize model from byte array."""
    bio = io.BytesIO(model_bytes)
    with h5py.File(bio) as f:
        return load_model_fn(f, custom_objects=CUSTOM_OBJECTS)

model = deserialize_model(best_model_bytes, tf.keras.models.load_model)
input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]
print(input_names, output_names)
model.summary()

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# frozen_graph = freeze_session(
    # K.get_session(),
    # output_names=output_names)
# tf.train.write_graph(frozen_graph, "/tmp/", "my_model.pb", as_text=False)

# model.save('saved_model', save_format='tf')

converter = trt.TrtGraphConverter(input_graph_def=frozen_graph)
frozen_graph = converter.convert()
# Failed to import metagraph, check error log for more info
