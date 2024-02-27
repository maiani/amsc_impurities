import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.optimize import linear_sum_assignment

def sort_eigensystem(ws, vs):
    """Sort the eigensystem of an Hamiltonian.
    
    Parameters:
    -----------
    ws : eigenvalues
    vs : eigenvectos
    
    Returns:
    ws_sorted : sorted eigenvalues
    vs_sorted : sorted eigenvectors
    """
    
    def best_match(psi1, psi2, threshold=None):
        """Find the best match of two sets of eigenvectors.

        Parameters:
        -----------
        psi1, psi2 : numpy 2D complex arrays
            Arrays of initial and final eigenvectors.
        threshold : float, optional
            Minimal overlap when the eigenvectors are considered belonging to the same band.
            The default value is :math:`1/(2N)^{1/4}`, where :math:`N` is the length of each eigenvector.

        Returns:
        --------
        sorting : numpy 1D integer array
            Permutation to apply to ``psi2`` to make the optimal match.
        diconnects : numpy 1D bool array
            The levels with overlap below the ``threshold`` that should be considered disconnected.
        """
        if threshold is None:
            threshold = (2 * psi1.shape[0]) ** -0.25
        Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
        orig, perm = linear_sum_assignment(-Q)
        return perm, Q[orig, perm] < threshold

    N = ws.shape[0]

    e = ws[0]
    psi = vs[0]

    ws_sorted = [e]
    vs_sorted = [psi]

    for i in range(1, N):
        e2 = ws[i]
        psi2 = vs[i]
        perm, line_breaks = best_match(psi, psi2)
        e2 = e2[perm]
        intermediate = (e + e2) / 2
        intermediate[line_breaks] = None
        psi = psi2[:, perm]
        e = e2

        ws_sorted.append(intermediate)
        ws_sorted.append(e)
        vs_sorted.append(psi)

    return np.array(ws_sorted)[::2], np.array(vs_sorted)



def apply_sw(
    v0, v, w, d, H: np.ndarray = None, method="svd",
):
    """
    Calculate the effective low-energy Hamiltonian with the Schrieffer-Wolff transformation.
    
        Parameters:
            - v0 : unperturbed system eigenvectors
            - v  : eigenvectors of the full system 
            - w  : eigenvalues of the full system 
            - H  : Hamiltonian (needed only for method "sqrt")
            - d  : desired dimension of the effective Hamiltonian
            - method : "sqrt" for matrix square root method and "svd" for SVD version
                         
        Returns:
            - H_eff : effective Hamiltonian in the v0 basis
            - U_sw  : Schrieffer-Wolff unitary (only for "sqrt" method)
    """

    assert v0.shape[-1] >= d
    assert v.shape[-1] >= d

    if H is not None:
        assert H.shape[0] == v.shape[-2]
        assert H.shape[1] == v.shape[-2]

    U_sw = None

    if method == "svd":
        P0 = v0[:, :d].conj().T
        P = v[:, :d]

        B = P0 @ P

        U, s, Vh = la.svd(B)
        A = U @ Vh

        H_eff = A @ np.diag(w[:d]) @ A.conj().T

    elif method == "sqrt":
        # TODO :  Rewrite P0 in the eigenvalue base to make it faster
        P0 = v0[:, :d] @ v0[:, :d].conj().T
        P = v[:, :d] @ v[:, :d].conj().T

        I = np.eye(H.shape[0])

        U_sw = la.sqrtm((2 * P0 - I) @ (2 * P - I))

        H_sw_full = U_sw @ H @ U_sw.conj().T
        H_eff = v0[:, :d].conj().T @ H_sw_full @ v0[:, :d]

    else:
        raise ValueError("""Method should be "sqrt" or "svd".""")

    return H_eff, U_sw

