import math
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ====== CONFIG ======
st.set_page_config(
    page_title="Numerica Studio",
    page_icon="📘",
    layout="wide"
)

# ====== CUSTOM STYLE ======
st.markdown("""
<style>

/* Ẩn thanh trắng */
.block-container {
    padding-top: 1.5rem;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Title */
.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #60a5fa;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    margin-bottom: 25px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* Button đẹp */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
}

/* Input */
.stTextInput>div>div>input {
    border-radius: 10px;
    height: 45px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

/* Hide footer */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ====== HEADER ======
st.markdown('<div class="main-title">📘 Numerica Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Scientific Computing • Visualization • Matrix Decomposition</div>',
    unsafe_allow_html=True
)


# =========================================================
# SAFE MATH ENV
# =========================================================
SAFE_SCALAR_ENV = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,      # natural log
    "log10": math.log10,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
    "factorial": math.factorial,
}

SAFE_VECTOR_ENV = {
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
}


# =========================================================
# HELPERS
# =========================================================
def parse_matrix(text: str) -> np.ndarray:
    """
    Parse matrix from text:
    Example:
    1 2 3
    4 5 6
    """
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        values = [float(x) for x in line.replace(",", " ").split()]
        rows.append(values)

    if not rows:
        raise ValueError("Ma trận rỗng.")

    col_count = len(rows[0])
    if any(len(row) != col_count for row in rows):
        raise ValueError("Các hàng phải có cùng số cột.")

    return np.array(rows, dtype=float)


def matrix_to_text(M: np.ndarray, precision: int = 6) -> str:
    return np.array2string(
        M,
        precision=precision,
        suppress_small=True,
    )


def is_square(A: np.ndarray) -> bool:
    return A.ndim == 2 and A.shape[0] == A.shape[1]


def orthonormalize_columns(vectors: List[np.ndarray], tol: float = 1e-10) -> List[np.ndarray]:
    basis = []
    for v in vectors:
        w = v.astype(float).copy()
        for b in basis:
            w = w - np.dot(b, w) * b
        norm = np.linalg.norm(w)
        if norm > tol:
            basis.append(w / norm)
    return basis


def null_space_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis of Ker(A) using SVD numerically.
    Used to complete right-singular vectors for zero singular values.
    """
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(s > tol)
    ns = vh[rank:].T
    if ns.size == 0:
        return np.empty((A.shape[1], 0))
    basis = orthonormalize_columns([ns[:, i] for i in range(ns.shape[1])], tol=tol)
    if not basis:
        return np.empty((A.shape[1], 0))
    return np.column_stack(basis)


def gram_schmidt_qr(A: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical Gram-Schmidt QR for linearly independent columns.
    """
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    for j in range(n):
        v = A[:, j].astype(float).copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        if R[j, j] < tol:
            raise ValueError("Các cột của A không độc lập tuyến tính hoặc ma trận gần suy biến.")
        Q[:, j] = v / R[j, j]

    return Q, R


def eigen_decomposition_symmetric(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For symmetric matrix A = P D P^T
    """
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Ma trận phải đối xứng để dùng chéo hóa trực giao.")
    eigvals, eigvecs = np.linalg.eigh(A)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs


def svd_from_theory(A: np.ndarray, tol: float = 1e-10) -> Dict[str, np.ndarray]:
    """
    Slide-style SVD:
    1) Diagonalize A^T A = V Λ V^T
    2) sigma_i = sqrt(lambda_i)
    3) u_i = A v_i / sigma_i for sigma_i > 0
    4) Complete V for zero singular values using orthonormal basis of Ker(A)
    """
    ATA = A.T @ A
    eigvals, eigvecs = np.linalg.eigh(ATA)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals = np.clip(eigvals, 0.0, None)
    sigmas = np.sqrt(eigvals)

    positive_idx = [i for i, s in enumerate(sigmas) if s > tol]
    zero_idx = [i for i, s in enumerate(sigmas) if s <= tol]

    V_pos = eigvecs[:, positive_idx] if positive_idx else np.empty((A.shape[1], 0))
    sigma_pos = sigmas[positive_idx] if positive_idx else np.array([])

    U_cols = []
    for i, sigma in enumerate(sigma_pos):
        v_i = V_pos[:, i]
        u_i = (A @ v_i) / sigma
        U_cols.append(u_i / np.linalg.norm(u_i))

    U_pos = np.column_stack(U_cols) if U_cols else np.empty((A.shape[0], 0))

    # Complete V with orthonormal basis of ker(A) if needed
    V_zero = null_space_basis(A, tol=tol)
    if V_zero.shape[1] < len(zero_idx):
        # fallback to eigenvectors from zero eigenspace if needed
        zero_vectors = eigvecs[:, zero_idx] if zero_idx else np.empty((A.shape[1], 0))
        combined = []
        for i in range(V_zero.shape[1]):
            combined.append(V_zero[:, i])
        for i in range(zero_vectors.shape[1]):
            combined.append(zero_vectors[:, i])
        basis = orthonormalize_columns(combined, tol=tol)
        if basis:
            V_zero = np.column_stack(basis[:len(zero_idx)])
        else:
            V_zero = np.empty((A.shape[1], 0))

    V = np.column_stack([V_pos, V_zero]) if (V_pos.size or V_zero.size) else np.empty((A.shape[1], 0))

    # Build full U from numpy SVD for completion if necessary
    # This keeps reconstruction stable for rectangular matrices
    U_full_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=True)

    # Use theory-based positive singular vectors when available
    if U_pos.shape[1] > 0:
        U_full = U_full_np.copy()
        U_full[:, :U_pos.shape[1]] = U_pos
    else:
        U_full = U_full_np

    # Sigma rectangular
    m, n = A.shape
    Sigma = np.zeros((m, n), dtype=float)
    diag_len = min(m, n, len(sigmas))
    Sigma[:diag_len, :diag_len] = np.diag(sigmas[:diag_len])

    # Vt: prefer theory-based V for first n columns when possible
    if V.shape[1] == n:
        Vt = V.T
    else:
        Vt = Vt_np

    return {
        "ATA": ATA,
        "eigenvalues_ATA": eigvals,
        "singular_values": sigmas,
        "U": U_full,
        "Sigma": Sigma,
        "Vt": Vt,
        "reconstructed": U_full @ Sigma @ Vt,
    }


def pca_from_theory(X: np.ndarray, standardize: bool = False) -> Dict[str, np.ndarray]:
    """
    PCA following slide flow:
    1) center or z-score standardize
    2) S = 1/(n-1) X^T X
    3) eigen-decompose S = P D P^T
    4) PC = X P
    """
    if X.ndim != 2:
        raise ValueError("Dữ liệu phải là ma trận 2 chiều (n_samples x n_features).")

    mean = X.mean(axis=0)
    X_centered = X - mean

    if standardize:
        std = X_centered.std(axis=0, ddof=1)
        std = np.where(np.abs(std) < 1e-12, 1.0, std)
        X_processed = X_centered / std
    else:
        std = None
        X_processed = X_centered

    n_samples = X_processed.shape[0]
    if n_samples < 2:
        raise ValueError("Cần ít nhất 2 quan sát để tính PCA.")

    S = (X_processed.T @ X_processed) / (n_samples - 1)
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    principal_components = X_processed @ eigvecs
    explained_ratio = eigvals / eigvals.sum() if eigvals.sum() > 0 else np.zeros_like(eigvals)
    cumulative_ratio = np.cumsum(explained_ratio)

    return {
        "mean": mean,
        "std": std,
        "X_processed": X_processed,
        "covariance": S,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "principal_components": principal_components,
        "explained_ratio": explained_ratio,
        "cumulative_ratio": cumulative_ratio,
    }


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Navigation")
mode = st.sidebar.radio(
    "Chọn module",
    [
        "Scientific Calculator",
        "Function Plotter",
        "Matrix Algebra",
        "Matrix Decomposition",
        "PCA Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Gợi ý nhập ma trận:\n\n"
    "1 2 3\n"
    "4 5 6\n"
    "7 8 9"
)

# =========================================================
# 1) SCIENTIFIC CALCULATOR
# =========================================================
if mode == "Scientific Calculator":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Scientific Calculator")

    expr = st.text_input(
        "Nhập biểu thức",
        value="sin(pi/2) + sqrt(16) + log(e)",
        help="Dùng ** cho lũy thừa. Ví dụ: (2+3)*5, sin(pi/4), factorial(5), log10(100).",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Compute", use_container_width=True):
            try:
                expr_clean = expr.replace("^", "**")
                result = eval(expr_clean, {"__builtins__": {}}, SAFE_SCALAR_ENV)
                st.success(f"Kết quả: {result}")
            except Exception as e:
                st.error(f"Lỗi khi tính biểu thức: {e}")

    with col2:
        if st.button("Clear Hint", use_container_width=True):
            st.info("Bạn có thể dùng: sin, cos, tan, sqrt, log, log10, exp, factorial, pi, e.")

    st.caption("Calculator hỗ trợ các phép tính khoa học cơ bản theo cú pháp Python.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 2) FUNCTION PLOTTER
# =========================================================
elif mode == "Function Plotter":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Function Plotter")

    tab2d, tab3d = st.tabs(["2D Function y = f(x)", "3D Surface z = f(x, y)"])

    with tab2d:
        c1, c2, c3 = st.columns(3)
        with c1:
            f2d = st.text_input("f(x)", value="sin(x) + 0.1*x**2")
        with c2:
            xmin = st.number_input("x min", value=-10.0)
        with c3:
            xmax = st.number_input("x max", value=10.0)

        if st.button("Plot 2D", use_container_width=True):
            try:
                if xmin >= xmax:
                    raise ValueError("x min phải nhỏ hơn x max.")

                x = np.linspace(xmin, xmax, 1000)
                env = SAFE_VECTOR_ENV.copy()
                env["x"] = x
                y = eval(f2d.replace("^", "**"), {"__builtins__": {}}, env)

                fig, ax = plt.subplots(figsize=(9, 4.8))
                ax.plot(x, y, linewidth=2)
                ax.set_title(f"y = {f2d}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Lỗi khi vẽ 2D: {e}")

    with tab3d:
        c1, c2 = st.columns(2)
        with c1:
            f3d = st.text_input("f(x, y)", value="sin(x) + cos(y)")
        with c2:
            grid_n = st.slider("Grid size", min_value=20, max_value=120, value=60)

        c3, c4 = st.columns(2)
        with c3:
            xy_min = st.number_input("Domain min", value=-5.0)
        with c4:
            xy_max = st.number_input("Domain max", value=5.0)

        if st.button("Plot 3D", use_container_width=True):
            try:
                if xy_min >= xy_max:
                    raise ValueError("Domain min phải nhỏ hơn domain max.")

                x = np.linspace(xy_min, xy_max, grid_n)
                y = np.linspace(xy_min, xy_max, grid_n)
                X, Y = np.meshgrid(x, y)

                env = SAFE_VECTOR_ENV.copy()
                env["x"] = X
                env["y"] = Y
                Z = eval(f3d.replace("^", "**"), {"__builtins__": {}}, env)

                fig = plt.figure(figsize=(9, 6))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.92)
                ax.set_title(f"z = {f3d}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Lỗi khi vẽ 3D: {e}")

    st.caption("Hỗ trợ nhiều hàm phổ biến: sin, cos, tan, log, log10, sqrt, exp, abs, pi, e.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 3) MATRIX ALGEBRA
# =========================================================
elif mode == "Matrix Algebra":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Matrix Algebra")

    colA, colB = st.columns(2)
    with colA:
        A_text = st.text_area(
            "Matrix A",
            value="1 2\n3 4",
            height=180,
        )
    with colB:
        B_text = st.text_area(
            "Matrix B",
            value="5 6\n7 8",
            height=180,
        )

    try:
        A = parse_matrix(A_text)
        B = parse_matrix(B_text)

        st.markdown("**Preview**")
        p1, p2 = st.columns(2)
        with p1:
            st.write("A =", A)
        with p2:
            st.write("B =", B)

        r1, r2, r3 = st.columns(3)

        if r1.button("A + B", use_container_width=True):
            if A.shape != B.shape:
                st.error("A và B phải cùng kích thước để cộng.")
            else:
                st.write(A + B)

        if r2.button("A - B", use_container_width=True):
            if A.shape != B.shape:
                st.error("A và B phải cùng kích thước để trừ.")
            else:
                st.write(A - B)

        if r3.button("A × B", use_container_width=True):
            if A.shape[1] != B.shape[0]:
                st.error("Số cột của A phải bằng số hàng của B.")
            else:
                st.write(A @ B)

        r4, r5, r6 = st.columns(3)

        if r4.button("det(A)", use_container_width=True):
            if not is_square(A):
                st.error("A phải là ma trận vuông.")
            else:
                st.write(np.linalg.det(A))

        if r5.button("inv(A)", use_container_width=True):
            if not is_square(A):
                st.error("A phải là ma trận vuông.")
            else:
                try:
                    st.write(np.linalg.inv(A))
                except np.linalg.LinAlgError:
                    st.error("A không khả nghịch.")

        if r6.button("Aᵀ", use_container_width=True):
            st.write(A.T)

        r7, r8, r9 = st.columns(3)

        if r7.button("rank(A)", use_container_width=True):
            st.write(np.linalg.matrix_rank(A))

        if r8.button("trace(A)", use_container_width=True):
            if not is_square(A):
                st.error("A phải là ma trận vuông.")
            else:
                st.write(np.trace(A))

        if r9.button("A⁺ (Pseudo-inverse)", use_container_width=True):
            st.write(np.linalg.pinv(A))

    except Exception as e:
        st.error(f"Lỗi ma trận: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 4) MATRIX DECOMPOSITION
# =========================================================
elif mode == "Matrix Decomposition":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Matrix Decomposition")

    A_text = st.text_area(
        "Nhập ma trận A",
        value="1 2\n3 4\n5 6",
        height=220,
    )

    try:
        A = parse_matrix(A_text)
        st.write("A =", A)

        decomp_mode = st.selectbox(
            "Chọn phân rã",
            ["QR (Gram-Schmidt)", "SVD (Theory-based)", "Eigen Decomposition (Symmetric)"],
        )

        if decomp_mode == "QR (Gram-Schmidt)":
            if st.button("Run QR", use_container_width=True):
                try:
                    Q, R = gram_schmidt_qr(A)
                    st.markdown("**Q**")
                    st.write(Q)
                    st.markdown("**R**")
                    st.write(R)
                    st.markdown("**Kiểm tra QᵀQ**")
                    st.write(Q.T @ Q)
                    st.markdown("**Kiểm tra QR**")
                    st.write(Q @ R)
                except Exception as e:
                    st.error(f"Lỗi QR: {e}")

        elif decomp_mode == "SVD (Theory-based)":
            if st.button("Run SVD", use_container_width=True):
                try:
                    result = svd_from_theory(A)

                    st.markdown("**AᵀA**")
                    st.write(result["ATA"])

                    st.markdown("**Eigenvalues of AᵀA**")
                    st.write(result["eigenvalues_ATA"])

                    st.markdown("**Singular values**")
                    st.write(result["singular_values"])

                    st.markdown("**U**")
                    st.write(result["U"])

                    st.markdown("**Σ**")
                    st.write(result["Sigma"])

                    st.markdown("**Vᵀ**")
                    st.write(result["Vt"])

                    st.markdown("**Reconstruction UΣVᵀ**")
                    st.write(result["reconstructed"])

                    err = np.linalg.norm(A - result["reconstructed"])
                    st.success(f"Sai số tái tạo Frobenius: {err:.6e}")

                except Exception as e:
                    st.error(f"Lỗi SVD: {e}")

        else:
            if st.button("Run Eigen Decomposition", use_container_width=True):
                try:
                    eigvals, eigvecs = eigen_decomposition_symmetric(A)
                    D = np.diag(eigvals)
                    A_recon = eigvecs @ D @ eigvecs.T

                    st.markdown("**Eigenvalues**")
                    st.write(eigvals)

                    st.markdown("**Eigenvectors matrix P**")
                    st.write(eigvecs)

                    st.markdown("**Diagonal matrix D**")
                    st.write(D)

                    st.markdown("**Reconstruction PDPᵀ**")
                    st.write(A_recon)

                except Exception as e:
                    st.error(f"Lỗi chéo hóa trực giao: {e}")

        st.caption(
            "Module này triển khai QR bằng Gram-Schmidt, SVD theo hướng AᵀA trong slide, "
            "và chéo hóa trực giao cho ma trận đối xứng."
        )

    except Exception as e:
        st.error(f"Lỗi ma trận đầu vào: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 5) PCA ANALYSIS
# =========================================================
elif mode == "PCA Analysis":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("PCA Analysis")

    X_text = st.text_area(
        "Nhập dữ liệu X (mỗi hàng là một quan sát, mỗi cột là một đặc trưng)",
        value="150 50 60\n170 70 80",
        height=220,
    )

    standardize = st.checkbox("Chuẩn hóa z-score trước PCA", value=True)
    n_components = st.number_input("Số thành phần giữ lại", min_value=1, value=1, step=1)

    if st.button("Run PCA", use_container_width=True):
        try:
            X = parse_matrix(X_text)
            result = pca_from_theory(X, standardize=standardize)

            eigvals = result["eigenvalues"]
            eigvecs = result["eigenvectors"]
            PCs = result["principal_components"]
            explained = result["explained_ratio"]
            cumulative = result["cumulative_ratio"]

            if n_components > X.shape[1]:
                st.error("Số thành phần giữ lại không được vượt số đặc trưng.")
            else:
                st.markdown("**Mean vector**")
                st.write(result["mean"])

                if standardize:
                    st.markdown("**Standard deviation vector**")
                    st.write(result["std"])

                st.markdown("**Processed data matrix X**")
                st.write(result["X_processed"])

                st.markdown("**Covariance matrix S = (1/(n-1)) XᵀX**")
                st.write(result["covariance"])

                st.markdown("**Eigenvalues**")
                st.write(eigvals)

                st.markdown("**Eigenvectors (columns)**")
                st.write(eigvecs)

                st.markdown("**Principal Components = XU**")
                st.write(PCs)

                st.markdown("**Explained variance ratio**")
                st.write(explained)

                st.markdown("**Cumulative explained variance ratio**")
                st.write(cumulative)

                st.markdown(f"**Reduced data using first {n_components} component(s)**")
                Z_k = PCs[:, :n_components]
                st.write(Z_k)

                st.markdown("**Reconstruction from selected components**")
                U_k = eigvecs[:, :n_components]
                X_hat_processed = Z_k @ U_k.T

                if standardize:
                    X_hat = X_hat_processed * result["std"] + result["mean"]
                else:
                    X_hat = X_hat_processed + result["mean"]

                st.write(X_hat)

                if X.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    ax.plot(
                        np.arange(1, len(cumulative) + 1),
                        cumulative,
                        marker="o",
                        linewidth=2,
                    )
                    ax.set_title("Cumulative Explained Variance")
                    ax.set_xlabel("Number of components")
                    ax.set_ylabel("Cumulative ratio")
                    ax.set_ylim(0, 1.05)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Lỗi PCA: {e}")

    st.caption(
        "PCA ở đây bám đúng luồng lý thuyết: center / z-score → covariance → eigen-decomposition → principal components."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER (PROFESSIONAL)
# =========================================================
st.markdown("---")

st.markdown("""
<style>
.footer {
    text-align: center;
    padding: 20px;
    color: #94a3b8;
    font-size: 14px;
}
.footer strong {
    color: #e2e8f0;
}
.footer a {
    color: #60a5fa;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>

<div class="footer">
    <div><strong>Numerica Studio</strong></div>
    <div>Developed by <strong>Nguyễn Huy Hoàng</strong></div>
    <div>
        📧 <a href="mailto:hoang.nguyenhuy2024ll@hcmut.edu.vn">
        hoang.nguyenhuy2024ll@hcmut.edu.vn</a>
    </div>
    <div>📱 0393 875 279</div>
</div>
""", unsafe_allow_html=True)