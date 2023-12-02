import matplotlib.pyplot as plt
import numpy as np
import math
import time

import torch
from torch.utils.data import DataLoader

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

from alternative_convolution_pooling_circuit import conv_circuit2, pool_circuit2

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
        # qc = qc.compose(conv_circuit2(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        # qc = qc.compose(conv_circuit2(params[param_index : (param_index + 3)]), [q1, q2])
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
        # qc = qc.compose(pool_circuit2(params[param_index : (param_index + 3)]), [source, sink])
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
    
def train_torch_qcnn(qcnn_estimator, loss_fn, initial_weights, train_loader):
    torch_model = TorchConnector(qcnn_estimator, initial_weights)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.1)
    torch_model.train()
    loss_list = []
    epochs = 3
    for epoch in range(epochs):
        total_loss = []
        for bidx, (data, label) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            output = torch_model(data)
            loss_val = loss_fn(
                output.float(), 
                label.float()
            )
            loss_val.backward()
            optimizer.step()
            total_loss.append(loss_val.item())
            print(f"Training on epoch {epoch}", end="\r")
        print("\n")
        loss_list.append(sum(total_loss) / len(total_loss))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))
    return torch_model


def test_torch_qcnn(torch_model, loss_fn, test_loader, batch_size):
    total_loss = []
    with torch.no_grad():

        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            output = torch_model(data)
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss = loss_fn(output, target)
            total_loss.append(loss.item())

        print(
            "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
                sum(total_loss) / len(total_loss), correct / len(test_loader) / batch_size * 100
            )
        )
    
# def test_torch_qcnn(torch_qcnn, test_data, test_labels):
#     torch_test_data = torch.tensor(test_data, dtype=torch.float)
#     torch_qcnn.eval()
#     pred = []
#     for i in range(len(torch_test_data)):
#         output = torch_qcnn(torch_test_data[i])
#         pred.append(np.sign(output.detach().numpy())[0])
#     accuracy = sum(np.asarray(pred) == np.asarray(test_labels)) / len(test_labels)
#     print(f"Accuracy from the test data : {np.round(100 * accuracy, 2)}%")
    
# objective_func_vals = []
# def callback_graph(weights, obj_func_eval):
#     objective_func_vals.append(obj_func_eval)
    
# def plot_graph():
#     plt.rcParams["figure.figsize"] = (12, 6)
#     plt.title("Objective function value against iteration")
#     plt.xlabel("Iteration")
#     plt.ylabel("Objective function value")
#     plt.plot(range(len(objective_func_vals)), objective_func_vals)
#     plt.show()
    
# def train_qcnn(qnn_estimator, optimizer, callback_fn, data, labels):
#     classifier = NeuralNetworkClassifier(
#         qnn_estimator,
#         optimizer=optimizer,
#         callback=callback_fn
#     )
    
#     s = time.time()
#     classifier.fit(np.asarray(data), np.asarray(labels))
#     e = time.time()
    
#     print(f"Accuracy from the train data : {np.round(100 * classifier.score(data, labels), 2)}%")
#     print(f"Training time: {np.round(e - s, 2)}")
#     plot_graph()
#     return classifier

# def test_qcnn(classifier, test_data, test_labels):
#     # test_predictions = classifier.predict(test_data)
#     test_data_np = np.asarray(test_data)
#     test_labels_np = np.asarray(test_labels)
#     print(f"Accuracy from the test data : {np.round(100 * classifier.score(test_data_np, test_labels_np), 2)}%")

if __name__ == "__main__":
    
    max_qubits = 16
    feat_map = qiskit.circuit.library.ZFeatureMap(max_qubits)
    circ_qcnn, qcnn_estimator = make_qcnn(max_qubits, "QCNN", feat_map)
    
    image_h, image_w = 4, 4
    assert(image_h * image_w == max_qubits)
    data, labels = generate_dataset(1000, image_h, image_w, 4)
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.4
    )

    # initial_weights = 0.1 * (2 * algorithm_globals.random.random(qcnn_estimator.num_weights) - 1)
    zero_weights = np.zeros(qcnn_estimator.num_weights) 
    BATCH = 20
    train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(list(zip(test_data, test_labels)), batch_size=BATCH)
    
    loss_fn = torch.nn.MSELoss()
    torch_model = train_torch_qcnn(qcnn_estimator, loss_fn, zero_weights, train_loader)
    test_torch_qcnn(torch_model, loss_fn, test_loader, BATCH)
    