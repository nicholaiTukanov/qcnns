import matplotlib.pyplot as plt
import numpy as np
import math
import time

import torch

from IPython.display import clear_output

import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

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

# def pool_circuit(params):
#     target = qiskit.QuantumCircuit(2)
#     target.rz(-np.pi / 2, 1)
#     target.cx(1, 0)
#     target.rz(params[0], 0)
#     target.ry(params[1], 1)
#     target.cx(0, 1)
#     target.ry(params[2], 1)
#     return target
def pool_circuit(params):
    target = qiskit.QuantumCircuit(2)
    # Modified and added quantum gates
    target.h(0)  # Hadamard gate on qubit 0
    target.rx(params[0], 0)  # Rotation around X-axis for qubit 0
    target.rz(params[1], 1)  # Rotation around Z-axis for qubit 1
    target.cx(0, 1)  # CNOT gate with control qubit 0 and target qubit 1
    target.ry(params[2], 0)  # Rotation around Y-axis for qubit 0
    target.cx(1, 0)  # CNOT gate with control qubit 1 and target qubit 0

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
    
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # we decompose the circuit for the QNN to avoid additional data copying
    qcnn_estimator = EstimatorQNN(
        circuit=circ.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=qcnn.parameters,
    )
    
    return circ, qcnn_estimator
    
    
def train_torch_qcnn(qcnn_estimator, initial_weights, train_data, train_labels):
    torch_model = TorchConnector(qcnn_estimator, initial_weights)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.1)
    loss = torch.nn.MSELoss()
    torch_train_data = torch.tensor(train_data, dtype=torch.float)
    torch_train_labels = torch.tensor(train_labels, dtype=torch.float)
    torch_model.train()
    batch_size = 10
    for i in range(0, len(torch_train_data), batch_size):
        optimizer.zero_grad()
        output = torch_model(torch_train_data[i:i+batch_size])
        loss_val = loss(output, torch_train_labels[i:i+batch_size])
        loss_val.backward()
        optimizer.step()
        print(f"Loss value: {loss_val}")
    return torch_model
    
def test_torch_qcnn(torch_qcnn, test_data, test_labels):
    torch_test_data = torch.tensor(test_data, dtype=torch.float)
    # torch_test_labels = torch.tensor(test_labels, dtype=torch.float)
    torch_qcnn.eval()
    pred = []
    for i in range(len(torch_test_data)):
        output = torch_qcnn(torch_test_data[i])
        pred.append(np.sign(output.detach().numpy())[0])
    accuracy = sum(np.asarray(pred) == np.asarray(test_labels)) / len(test_labels)
    print(f"Accuracy from the test data : {np.round(100 * accuracy, 2)}%")
    
objective_func_vals = []
def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    
def plot_graph():
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()
    
def train_qcnn(qnn_estimator, optimizer, callback_fn, data, labels):
    classifier = NeuralNetworkClassifier(
        qnn_estimator,
        optimizer=optimizer,
        callback=callback_fn
    )
    
    s = time.time()
    classifier.fit(np.asarray(data), np.asarray(labels))
    e = time.time()
    
    print(f"Accuracy from the train data : {np.round(100 * classifier.score(data, labels), 2)}%")
    print(f"Training time: {np.round(e - s, 2)}")
    plot_graph()
    return classifier

def test_qcnn(classifier, test_data, test_labels):
    # test_predictions = classifier.predict(test_data)
    test_data_np = np.asarray(test_data)
    test_labels_np = np.asarray(test_labels)
    print(f"Accuracy from the test data : {np.round(100 * classifier.score(test_data_np, test_labels_np), 2)}%")

if __name__ == "__main__":
    
    max_qubits = 8
    feat_map = qiskit.circuit.library.ZFeatureMap(max_qubits)
    circ_qcnn, qcnn_estimator = make_qcnn(max_qubits, "QCNN", feat_map)
    # circ_qcnn.draw("mpl", 1, "qcnn.png")
    
    image_h, image_w = 4, 2
    assert(image_h * image_w == max_qubits)
    data, labels = generate_dataset(10000, image_h, image_w, 2)
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.3
    )
    
    # uncomment to print the first 4 images
    # fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
    # for i in range(4):
    #     ax[i // 2, i % 2].imshow(
    #         train_data[i].reshape(2, 4),  # Change back to 2 by 4
    #         aspect="equal",
    #     )
    # plt.subplots_adjust(wspace=0.1, hspace=0.025)
    # plt.show()
    
    # classifier = train_qcnn(
    #     qcnn_estimator, 
    #     qiskit_optimizers.COBYLA(maxiter=1000), 
    #     callback_graph, 
    #     train_data, train_labels
    # )
    
    # test_qcnn(classifier, test_data, test_labels)
    initial_weights = 0.1 * (2 * algorithm_globals.random.random(qcnn_estimator.num_weights) - 1)
    torch_model = train_torch_qcnn(qcnn_estimator, initial_weights, train_data, train_labels)
    test_torch_qcnn(torch_model, test_data, test_labels)
    