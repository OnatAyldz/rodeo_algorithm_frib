import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from tqdm import tqdm

def generate_superiteration_times(tsamples, superiterations, si_time):
    """Helper function to generate superiteration time parameters."""
    if len(si_time) < len(tsamples) * superiterations:
        raise ValueError("Insufficient si_time parameters for the given cycles and superiterations.")
    
    superiteration_parameter_binds = {}
    for i in range(len(tsamples)):
        superiteration_times = []
        time = tsamples[i]
        for _ in range(superiterations):
            time /= 2
            superiteration_times.append(time)
        
        for j in range(superiterations):
            si_index = i * superiterations + j
            superiteration_parameter_binds[si_time[si_index]] = superiteration_times[j]
    
    return superiteration_parameter_binds

def create_hopping_gate(J, delta_t):
    """Create the hopping interaction gate."""
    f_circ = QuantumCircuit(2)
    f_circ.h([0, 1])
    f_circ.s([0, 1])
    f_circ.h([0, 1])
    f_circ.cx(0, 1)
    f_circ.h(1)
    f_circ.z(1)
    f_circ.s([0, 1])
    f_circ.h([0, 1])
    f_gate = f_circ.to_gate(label='F_gate')
    
    g_circ = QuantumCircuit(2)
    g_circ.h([0, 1])
    g_circ.z(0)
    g_circ.s([0, 1])
    g_circ.h(1)
    g_circ.cx(0, 1)
    g_circ.h([0, 1])
    g_circ.z([0, 1])
    g_circ.s([0, 1])
    g_circ.h([0, 1])
    g_gate = g_circ.to_gate(label='G_gate')

    hop_circ = QuantumCircuit(2)
    hop_circ.append(f_gate, [0, 1])
    hop_circ.ry((J * delta_t), 0)
    hop_circ.ry((-1*J * delta_t), 1)
    hop_circ.append(g_gate, [0, 1])
    return hop_circ.to_gate(label=fr"A_gate_{J}")

def create_onsite_gate(U, delta_t):
    """Create the on-site interaction gate."""
    onsite_circ = QuantumCircuit(2)
    onsite_circ.cx(0, 1)
    onsite_circ.rz(2*U * delta_t, 1)
    onsite_circ.cx(0, 1)
    return onsite_circ.to_gate(label=fr"B_gate_{U}")

def rodeo_cycle(num_sites, J_value, U_value, t: Parameter, r, targ: Parameter):
    """Create a single rodeo cycle."""
    beta = t / r
    num_qubits = num_sites * 2

    sys = QuantumRegister(num_qubits, 's')
    aux = QuantumRegister(1, 'a')
    qc = QuantumCircuit(sys, aux)

    qc.h(aux[0])
    
    A_gate = create_hopping_gate(J=-J_value, delta_t=beta)
    B_gate = create_onsite_gate(U=U_value, delta_t=beta)

    qc.cz([sys[0], sys[1]], aux[0])
    for _ in range(r):
        # Hopping terms for spin-up qubits
        for site in range(0, num_sites - 1, 2):
            qc.append(A_gate, [site * 2, (site + 1) * 2])
        for site in range(1, num_sites - 1, 2):
            qc.append(A_gate, [site * 2, (site + 1) * 2])

        # Hopping terms for spin-down qubits
        for site in range(0, num_sites - 1, 2):
            qc.append(A_gate, [site * 2 + 1, (site + 1) * 2 + 1])
        for site in range(1, num_sites - 1, 2):
            qc.append(A_gate, [site * 2 + 1, (site + 1) * 2 + 1])

        # CX gates
        for i in range(0, num_qubits, 4):
            qc.cx(aux[0], sys[i])
            if i + 2 < num_qubits:
                qc.cx(aux[0], sys[i + 2])

        # On-site interactions
        for site in range(num_sites):
            qc.append(B_gate, [site * 2, site * 2 + 1])

        # Second set of CX gates
        for i in range(0, num_qubits, 4):
            qc.cx(aux[0], sys[i])
            if i + 2 < num_qubits:
                qc.cx(aux[0], sys[i + 2])
    
    qc.cz([sys[0], sys[1]], aux[0])
    qc.p(2*targ * t, aux[0])
    qc.h(aux[0])

    return qc

def create_rodeo_circuit(num_sites, J_input, U_input, cycles, iterations, steps=5):
    """
    Create a complete rodeo circuit with the specified parameters.
    
    Args:
        num_sites (int): Number of sites in the system
        J_input (float): Hopping parameter
        U_input (float): On-site interaction strength
        cycles (int): Number of rodeo cycles
        iterations (int): Number of iterations per cycle
        steps (int, optional): Number of Trotter steps. Defaults to 5.
    
    Returns:
        tuple: (QuantumCircuit, list of time Parameters, list of SI time Parameters, target Parameter)
    """
    target = Parameter(r'$E_\odot$')
    time = [Parameter(fr'$t_{i}$') for i in range(cycles)]
    si_time = [Parameter(fr'$st_{j}$') for j in range(cycles * iterations)]
    
    classical = ClassicalRegister(cycles * (1 + iterations), 'c')
    aux = QuantumRegister(1, 'a')
    sys = QuantumRegister(num_sites * 2, 's')
    circuit = QuantumCircuit(sys, aux, classical)
    
    # Initial state preparation
    circuit.x([sys[1], sys[2]])
    
    classical_idx = 0
    super_idx = 0
    
    for j in range(cycles):
        rodeo_gate = rodeo_cycle(num_sites=num_sites, J_value=J_input, U_value=U_input,
                                t=time[j], r=steps, targ=target)
        circuit.append(rodeo_gate.to_gate(label=fr'Rodeo_Cycle_{j}'), range(num_sites * 2 + 1))
        circuit.measure(aux, classical[classical_idx])
        classical_idx += 1
        
        for k in range(iterations):
            rodeo_gate_si = rodeo_cycle(num_sites=num_sites, J_value=J_input, U_value=U_input,
                                      t=si_time[super_idx], r=steps, targ=target)
            circuit.append(rodeo_gate_si.to_gate(label=fr'SI_Rodeo_Cycle_{j}_{k}'),
                         range(num_sites * 2 + 1))
            circuit.measure(aux, classical[classical_idx])
            classical_idx += 1
            super_idx += 1
    
    return circuit, time, si_time, target

def run_rodeo_simulation(circuit, time_params, si_time_params, target_param, 
                        energy_min, energy_max, delta_energy, gamma, 
                        timeresamples=10, shots_per_time=1024):
    """
    Run the rodeo simulation for a range of energies.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to run
        time_params (list): List of time Parameters
        si_time_params (list): List of superiteration time Parameters
        target_param (Parameter): Target energy Parameter
        energy_min (float): Minimum energy to scan
        energy_max (float): Maximum energy to scan
        delta_energy (float): Energy step size
        gamma (float): Gamma parameter for time sampling
        timeresamples (int, optional): Number of time samples. Defaults to 10.
        shots_per_time (int, optional): Number of shots per time sample. Defaults to 1024.
    
    Returns:
        tuple: (energies, probabilities)
    """
    cycles = len(time_params)
    iterations = len(si_time_params) // cycles
    print(f"Total expected measurements: {cycles * (1 + iterations)}")
    print(f"Cycles: {cycles}, Iterations: {iterations}")
    energies = np.arange(energy_min, energy_max + delta_energy, delta_energy)
    all_probabilities = []
    
    for energy in tqdm(energies, desc="Processing energies", ncols=100):
        targ_energy = {target_param: energy}
        probabilities_0 = []
        
        for _ in range(timeresamples):
            tsamples = ((1 / gamma) * np.random.randn(len(time_params))).tolist()
            time_parameters = dict(zip(time_params, tsamples))
            superiteration_parameters = generate_superiteration_times(
                tsamples, 
                superiterations=len(si_time_params)//len(time_params),
                si_time=si_time_params
            )
            
            # Assign parameters and run circuit
            circuit_bound = circuit.assign_parameters(time_parameters, inplace=False)
            circuit_bound = circuit_bound.assign_parameters(targ_energy, inplace=False)
            circuit_bound = circuit_bound.assign_parameters(superiteration_parameters, inplace=False)
            
            sampler = Sampler()
            result = sampler.run(circuit_bound, shots=shots_per_time).result()
            quasi_dists = result.quasi_dists
            
            for dist in quasi_dists:
                probabilities_0.append(dist.get(0, 0))
        
        avg_prob_0 = np.mean(probabilities_0)
        all_probabilities.append(avg_prob_0)
    
    return energies, all_probabilities

def assign_circuit_parameters(circuit, time_params, si_time_params, target_param, gamma, energy):
    """
    Assign given circuit parameters to circuit
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to run
        time_params (list): List of time Parameters
        si_time_params (list): List of superiteration time Parameters
        target_param (Parameter): Target energy Parameter
        gamma (float): Gamma parameter for time sampling
    
    Returns:
        tuple: (energies, probabilities)
    """

    tsamples = ((1 / gamma) * np.random.randn(len(time_params))).tolist()
    time_parameters = dict(zip(time_params, tsamples))
    superiteration_parameters = generate_superiteration_times(
        tsamples, 
        superiterations=len(si_time_params)//len(time_params),
        si_time=si_time_params
    )

    targ_energy = {target_param : energy}
    
    # Assign parameters and run circuit
    circuit_bound = circuit.assign_parameters(time_parameters, inplace=False)
    circuit_bound = circuit_bound.assign_parameters(targ_energy, inplace=False)
    circuit_bound = circuit_bound.assign_parameters(superiteration_parameters, inplace=False)

    return circuit_bound

