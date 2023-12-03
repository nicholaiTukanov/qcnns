from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np



def alternative_pool_circuit(params):
    target = QuantumCircuit(2)
    # Modified and added quantum gates
    target.h(0)  # Hadamard gate on qubit 0
    target.rx(params[0], 0)  # Rotation around X-axis for qubit 0
    target.rz(params[1], 1)  # Rotation around Z-axis for qubit 1
    target.cx(0, 1)  # CNOT gate with control qubit 0 and target qubit 1
    target.ry(params[2], 0)  # Rotation around Y-axis for qubit 0
    target.cx(1, 0)  # CNOT gate with control qubit 1 and target qubit 0

    return target

# params = ParameterVector("θ", length=3)
# circuit = alternative_pool_circuit(params)
# circuit.draw("mpl")

# def pool_layer(sources, sinks, param_prefix):
#     num_qubits = len(sources) + len(sinks)
#     qc = QuantumCircuit(num_qubits, name="Pooling Layer")
#     param_index = 0
#     params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
#     for source, sink in zip(sources, sinks):
#         qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
#         qc.barrier()
#         param_index += 3

#     qc_inst = qc.to_instruction()

#     qc = QuantumCircuit(num_qubits)
#     qc.append(qc_inst, range(num_qubits))
#     return qc


# sources = [0, 1]
# sinks = [2, 3]
# circuit = pool_layer(sources, sinks, "θ")
# circuit.decompose().draw("mpl")