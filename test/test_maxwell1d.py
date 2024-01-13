
from dgtd.maxwell1d import *
from dgtd.mesh1d import *

import matplotlib.pyplot as plt

def test_get_energy_N1():
    m = Mesh1D(0, 1, 10)
    sp = Maxwell1D(1, m)
    fields = sp.buildFields()
    
    fields['E'].fill(0.0)
    fields['E'][0,0] = 1.0
    assert np.isclose(sp.getEnergy(fields['E']), 0.1*1.0/3.0, rtol=1e-9)

    fields['E'].fill(1.0)
    assert np.isclose(sp.getEnergy(fields['E']),         1.0, rtol=1e-9)

def test_buildEvolutionOperator():
    m = Mesh1D(0, 1, 10)
    sp = Maxwell1D(2, m, "Centered")
    A = sp.buildEvolutionOperator()
    eigA, _ = np.linalg.eig(A)

    assert A.shape == (60, 60)
    assert np.allclose(np.real(eigA), 0)

def test_buildEvolutionOperator_sorting():
    m = Mesh1D(0, 1, 3)
    sp = Maxwell1D(2, m, "Centered")
    Np = sp.number_of_nodes_per_element()
    K = m.number_of_elements()
    
    A = sp.buildEvolutionOperator()

    # import matplotlib.pyplot as plt
    # plt.spy(A)
    # plt.show()