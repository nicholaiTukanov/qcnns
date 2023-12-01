import matplotlib.pyplot as plt
import numpy as np
import math
import time

import torch
from torch.utils.data import DataLoader

import qiskit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from sklearn.model_selection import train_test_split

from img_gen import generate_dataset


# source: qiskit
# *****************************************************************************
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

# *****************************************************************************
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

# *****************************************************************************
def pool_circuit(params):
    target = qiskit.QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

# *****************************************************************************
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

# *****************************************************************************
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

# *****************************************************************************
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