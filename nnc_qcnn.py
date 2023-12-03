import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pickle

import torch
import torch.nn as nn

import qiskit
from qiskit.utils import QuantumInstance
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator, BackendEstimator

import qiskit_algorithms.optimizers as qiskit_optimizers

from sklearn.model_selection import train_test_split

from img_gen import generate_dataset


# using qiskit tutorial to build a convolutional quantum circuit
def conv_circuit(params):
    circ = qiskit.QuantumCircuit(2)
    circ.rz(-np.pi / 2, 1)
    circ.cx(1, 0)
    circ.rz(params[0], 0)
    circ.ry(params[1], 1)
    circ.cx(0, 1)
    circ.ry(params[2], 1)
    circ.cx(1, 0)
    circ.rz(np.pi / 2, 0)
    return circ

def conv_layer(num_qubits, param_prefix):
    qc = qiskit.QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = qiskit.circuit.ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = qiskit.QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = qiskit.QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = qiskit.circuit.ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()

    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def make_qcnn(num_qubits, name, feature_map):
    # assumes num_qubits is a power of 2
    total_layers = math.log2(num_qubits)
    qcnn = qiskit.QuantumCircuit(num_qubits, name=name)
    cur_qubits = num_qubits
    conv_idx = 0
    for i in range(int(total_layers)):
        
        qubit_idx = list(range(conv_idx, num_qubits))
        qcnn.compose(
            conv_layer(cur_qubits, "conv" + str(i)), 
            qubit_idx, 
            inplace=True
        )
        
        qcnn.compose(
            pool_layer(list(range(cur_qubits // 2)), list(range(cur_qubits // 2, cur_qubits)), "pool" + str(i)),
            qubit_idx,
            inplace=True
        )
        
        cur_qubits = cur_qubits // 2
        conv_idx += cur_qubits
    circ = qiskit.QuantumCircuit(num_qubits)
    circ.compose(feature_map, range(num_qubits), inplace=True)
    circ.compose(qcnn, range(num_qubits), inplace=True)

    # we decompose the circuit for the QNN to avoid additional data copying
    qcnn_estimator = EstimatorQNN(
        circuit=circ.decompose(),
        input_params=feature_map.parameters,
        weight_params=qcnn.parameters,
    )
    
    return circ, qcnn_estimator
    
objective_func_vals = []
def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    
# def plot_graph():
#     plt.rcParams["figure.figsize"] = (12, 6)
#     plt.title("Objective function value against iteration")
#     plt.xlabel("Iteration")
#     plt.ylabel("Objective function value")
#     plt.plot(range(len(objective_func_vals)), objective_func_vals)
#     plt.show()
    
obj_fn_vals = {}
def train_qcnn(qnn_estimator, optimizer, callback_fn, data, labels, init_weights, init_name):
    
    if init_weights is None:
        init_weights = np.random.rand(qnn_estimator.num_weights)
        
    classifier = NeuralNetworkClassifier(
        qnn_estimator,
        loss="cross_entropy",
        optimizer=optimizer,
        callback=callback_fn,
        initial_point=init_weights
    )
    
    s = time.time()
    classifier.fit(np.asarray(data), np.asarray(labels))
    e = time.time()
    
    print(f"Accuracy from the train data : {np.round(100 * classifier.score(data, labels), 2)}%")
    print(f"Training time: {np.round(e - s, 2)}")
    obj_fn_vals[init_name] = np.array(objective_func_vals).copy()
    objective_func_vals.clear()
    return classifier

def test_qcnn(classifier, test_data, test_labels):
    test_data_np = np.asarray(test_data)
    test_labels_np = np.asarray(test_labels)
    print(f"Accuracy from the test data : {np.round(100 * classifier.score(test_data_np, test_labels_np), 2)}%")

if __name__ == "__main__":
    
    max_qubits = 8
    image_h, image_w = 2, 4
    assert(image_h * image_w == max_qubits)
    data, labels = generate_dataset(300, image_h, image_w, 2)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3)
    print(f"Generated dataset with {len(data)} images")
    
    feat_map = qiskit.circuit.library.ZFeatureMap(max_qubits)
    
    circ_qcnn, qcnn_estimator = make_qcnn(max_qubits, "QCNN", feat_map)
    print(f"Made QCNN using {max_qubits} qubits")
    
    init_weights_zeros = np.zeros(qcnn_estimator.num_weights)
    init_weights_ones = np.ones(qcnn_estimator.num_weights)
    init_weights_xaiver = nn.init.xavier_uniform_(torch.empty(qcnn_estimator.num_weights,1)).numpy().flatten()
    init_weights_constant = nn.init.constant_(torch.empty(qcnn_estimator.num_weights), 1/qcnn_estimator.num_weights).numpy()
    init_weights_uniform = nn.init.uniform_(torch.empty(qcnn_estimator.num_weights)).numpy()
    init_weights_normal = nn.init.normal_(torch.empty(qcnn_estimator.num_weights)).numpy()
    
    init_weights = {
        "zeros": init_weights_zeros, 
        "ones": init_weights_ones,
        "xavier": init_weights_xaiver,
        "constant": init_weights_constant,
        "uniform": init_weights_uniform, 
        "normal": init_weights_normal
    }
    
    for name, weights in init_weights.items():
        print(f"Training with {name} weights")
        qcnn_classifier = train_qcnn(
            qcnn_estimator, 
            qiskit_optimizers.COBYLA(maxiter=200, rhobeg=0.5), 
            callback_graph,
            train_data, train_labels, 
            weights,
            name
        )
        print(f"Trained with {name} weights")
        test_qcnn(qcnn_classifier, test_data, test_labels)
    
    optimizer = "COBYLA"
    loss = "Cross Entropy"
    with open(f"obj_fn_vals_{optimizer}_{loss}_2.pickle", "wb") as f:
        pickle.dump(obj_fn_vals, f)
    
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.title(f"{loss} values using {optimizer} over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    x = range(len(obj_fn_vals["zeros"]))
    for name, obj_fn_val in obj_fn_vals.items():
        plt.plot(x, obj_fn_val, label=name)
    plt.legend()
    plt.show()
