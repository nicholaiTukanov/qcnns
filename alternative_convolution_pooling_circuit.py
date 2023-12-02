from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np

def conv_circuit2(params):
    target = QuantumCircuit(2)
    target.h(0)  # Hadamard gate for superposition
    target.cx(0, 1)  # CNOT gate for entanglement
    target.rz(params[0], 0)  # RZ rotation on qubit 0
    target.ry(params[1], 0)  # RY rotation on qubit 0
    target.rz(params[2], 1)  # RZ rotation on qubit 1
    target.cx(1, 0)  # Another CNOT for further entanglement
    target.h(1)  # Hadamard gate on qubit 1
    return target

def pool_circuit2(params):
    target = QuantumCircuit(2)
    # Apply a series of gates that manipulate and entangle the qubits
    target.cx(0, 1)  # Entangling qubits
    target.rz(params[0], 0)  # Rotation around Z-axis on qubit 0
    target.h(1)  # Hadamard gate on qubit 1 for superposition
    target.cx(1, 0)  # Further entangling qubits
    target.rz(params[1], 1)  # Another rotation on qubit 1
    target.h(0)  # Hadamard gate on qubit 0
    # Optionally, measure one qubit here if you want to 'pool' by measurement
    return target

def pool_circuit3(params):
    target = QuantumCircuit(2)
    # Modified and added quantum gates
    target.h(0)  # Hadamard gate on qubit 0
    target.rx(params[0], 0)  # Rotation around X-axis for qubit 0
    target.rz(params[1], 1)  # Rotation around Z-axis for qubit 1
    target.cx(0, 1)  # CNOT gate with control qubit 0 and target qubit 1
    target.ry(params[2], 0)  # Rotation around Y-axis for qubit 0
    target.cx(1, 0)  # CNOT gate with control qubit 1 and target qubit 0

    return target
