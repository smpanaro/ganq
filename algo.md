3.1 OPTIMIZATION MODEL FOR NON-UNIFORM QUANTIZATION
Consider a linear layer with weight matrix $\mathbf{W}\in\mathbb{R}^{m\times n}$ and input activation $\mathbf{X}\in\mathbb{R}^{n\times p}$. As shown in the right part of Figure 1(a) [https://arxiv.org/html/2501.12956v2#S1.F1.sf1], LUT-based quantization aims to compress $\mathbf{W}$ by representing its elements using a codebook. Specifically, the elements of the $i$-th channel in the quantized weight matrix $\mathbf{\widetilde{W}}$ are selected from the codebook $\mathbf{T}_{i}=\{t_{i,0},t_{i,1},\dots,t_{i,2^{N}-1}\}$, where $N$ is the bit-width of the quantization (e.g., 3 or 4 bits). Thus, each element $\mathbf{\widetilde{W}}_{i,j}$ satisfies $\mathbf{\widetilde{W}}_{i,j}\in\mathbf{T}_{i}$.
In practice, LUT-based quantization stores two components: a low-bit query matrix $\mathbf{Q}\in\{0,1,\dots 2^{N}-1\}^{m\times n}$, which specifies the indices of values in the codebook, and the codebook itself, $\mathbf{T}\in\mathbb{R}^{m\times 2^{N}}$, which contains the quantized values for each channel. For example, if $\mathbf{Q}_{ij}=0$, then $\mathbf{\widetilde{W}}_{i,j}=t_{i,0}$. Compared to the widely used basic per-channel uniform quantization (Frantar et al., 2022 [https://arxiv.org/html/2501.12956v2#bib.bib10]; Xiao et al., 2023 [https://arxiv.org/html/2501.12956v2#bib.bib38]), which requires two parameters per channel (i.e., scale and zero-point), this mechanism demands slightly more storage. However, as $\min\{m,n\}\gg 2^{N}$ in practice, the additional storage overhead is negligible. As shown in Table 1 [https://arxiv.org/html/2501.12956v2#S3.T1], for typical model sizes, the storage usage of LUT-based quantization remains comparable to the basic uniform quantization, differing by less than 0.2%. Moreover, some uniform quantization methods, such as OmniQuant (Shao et al., 2024 [https://arxiv.org/html/2501.12956v2#bib.bib32]) and AffineQuant (Ma et al., 2024 [https://arxiv.org/html/2501.12956v2#bib.bib23]), also require extra parameters.
To enable effective LUT-based non-uniform quantization, we formulate an optimization model aimed at minimizing the layer-wise output error.:
$\min_{\mathbf{Q},\mathbf{T}}\|\mathbf{W}\mathbf{X}-\mathbf{\widetilde{W}}%
\mathbf{X}\|_{F}^{2},\;s.t.\;\mathbf{\widetilde{W}}_{i,j}=\mathbf{T}_{i,%
\mathbf{Q}_{i,j}},\;\forall i,j,$
(1)
where $\|\cdot\|_{F}$ denotes the Frobenius norm, and $\mathbf{Q}$ and $\mathbf{T}$ are the decision variables.
Note that the quantized output for each row $(\mathbf{\widetilde{W}}\mathbf{X})_{i,:}$ depends only on its corresponding codebook $\mathbf{T}_{i,:}$ and query vector $\mathbf{Q}_{i,:}$. Consequently, the model in (1 [https://arxiv.org/html/2501.12956v2#S3.E1]) is inherently decomposable across the rows of $\mathbf{W}$. Leveraging this property, the problem can be reformulated into $m$ independent subproblems, which are highly parallelizable and particularly suitable for GPU acceleration. Specifically, this parallelization is achieved by expressing computations in matrix form, which enables efficient matrix-vector and element-wise operations across rows. Furthermore, each subproblem can be expressed as a mixed-integer quadratic programming problem:
$\min_{\mathbf{S}_{i},\mathbf{T}_{i}}\|\mathbf{W}_{i}\mathbf{X}-\mathbf{T}_{i}%
\mathbf{S}_{i}\mathbf{X}\|^{2}\;s.t.\;\mathbf{1}^{\top}\mathbf{S}_{i}=\mathbf{%
1}^{\top},\;\forall i,$
(2)
where $\mathbf{W}_{i}\in\mathbb{R}^{1\times n}$ is the $i$-th row of $\mathbf{W}$, $\mathbf{T}_{i}\in\mathbb{R}^{1\times 2^{N}}$ is the $i$-th row of $\mathbf{T}$, $\mathbf{S}_{i}\in\{0,1\}^{2^{N}\times n}$ is a column-wise one-hot encoding matrix indicating the mapping of elements from $\mathbf{T}_{i}$, and $\mathbf{1}$ denotes an all-one vector.
The mixed-integer structure of $\mathbf{S}_{i}$ introduces significant combinatorial complexity, and the bilinear interaction between $\mathbf{S}_{i}$ and $\mathbf{T}_{i}$ in the objective further compounds the computational challenge, rendering the problem inherently non-convex and non-smooth. These factors pose serious difficulties for off-the-shelf solvers, especially in large-scale settings with high-dimensional weight matrices and input activations. In response, we develop a specialized, GPU-adaptive approach tailored to navigate this complex search space while scaling to practical problem sizes.
3.2 GPU-ADAPTIVE NON-UNIFORM QUANTIZATION METHOD
To efficiently solve the model in (2 [https://arxiv.org/html/2501.12956v2#S3.E2]) for LUT-based non-uniform quantization, we employ an alternating direction optimization framework. This framework iteratively updates $\mathbf{S}_{i}$ and $\mathbf{T}_{i}$ by decomposing the objective into two subproblems. Each subproblem optimizes one decision variable while keeping the other fixed. The iterative scheme is outlined as follows:
$\displaystyle\mathbf{S}_{i}^{k+1}\!=\!\operatorname*{argmin}_{\mathbf{S}_{i}}%
\!\left\{\|\mathbf{W}_{i}\mathbf{X}-\mathbf{T}_{i}^{k}\mathbf{S}_{i}\mathbf{X}%
\|^{2}\;|\;\mathbf{1}^{\top}\mathbf{S}_{i}\!=\!\mathbf{1}^{\top}\!\right\}\!\!,$
(3)
$\displaystyle\mathbf{T}_{i}^{k+1}\!=\!\operatorname*{argmin}_{\mathbf{T}_{i}}%
\!\left\{\|\mathbf{W}_{i}\mathbf{X}-\mathbf{T}_{i}\mathbf{S}_{i}^{k+1}\mathbf{%
X}\|^{2}\right\}\!\!.$
(4)
The $\mathbf{T}_{i}$-subproblem in (4 [https://arxiv.org/html/2501.12956v2#S3.E4]) is an unconstrained quadratic program and admits a closed form solution given by:
$\mathbf{T}_{i}^{k+1}\!=\!\mathbf{W}_{i}\mathbf{XX}^{\top}\!(\mathbf{S}_{i}^{k+%
1})^{\top}\!((\mathbf{S}_{i})^{k+1}\mathbf{XX}^{\top}(\mathbf{S}_{i}^{k+1})^{%
\top})^{\dagger},$
(5)
where $(\cdot)^{\dagger}$ denotes the Moore-Penrose inverse.
Notably, the matrix $(\mathbf{S}_{i})^{k+1}\mathbf{XX}^{\top}(\mathbf{S}_{i}^{k+1})^{\top}$ has dimensions $2^{N}\times 2^{N}$, which is relatively small in practice (e.g., $16\times 16$ under 4-bit quantization), ensuring that the computation remains efficient. Moreover, computing (5 [https://arxiv.org/html/2501.12956v2#S3.E5]) involves only matrix-vector multiplications, making it highly efficient for GPU acceleration.
Since the solutions to all $\mathbf{T}_{i}$-subproblems share the same formulation, they can be combined into a single batch computation by stacking all $\mathbf{W}_{i}$ and $\mathbf{T}_{i}$ vectors row-wise and organizing $\mathbf{S}_{i}$ matrices into a tensor. Then, matrix operations can be used to efficiently compute the batch. This approach leverages modern GPUs’ parallel processing capabilities, significantly reducing computational overhead and improving overall efficiency.
Refer to caption [x3.png]
Figure 2: An illustration of the back-substitution framework for determining $\mathbf{S}_{i}$, leveraging the lower triangular structure of $\mathbf{L}$.
The primary challenge lies in the $\mathbf{S}_{i}$-subproblem (3 [https://arxiv.org/html/2501.12956v2#S3.E3]), which is a discrete, non-convex, and non-smooth combinatorial optimization problem. In the case of 4-bit quantization, each element of $\mathbf{S}_{i}$ can assume one of 16 possible values. A brute-force search over all combinations would require $\mathcal{O}(16^{n})$ operations, rendering it computationally prohibitive. Therefore, developing efficient solution techniques is essential for practical applications.
To address the $\mathbf{S}_{i}$-subproblem, we propose an efficient method that leverages the problem’s inherent structure. The objective in (3 [https://arxiv.org/html/2501.12956v2#S3.E3]) can be expanded as:
$\displaystyle\|\mathbf{W}_{i}\mathbf{X}-\mathbf{T}_{i}^{k}\mathbf{S}_{i}%
\mathbf{X}\|^{2}$
(6)
$\displaystyle=$
$\displaystyle(\mathbf{W}_{i}-\mathbf{T}_{i}^{k}\mathbf{S}_{i})(\mathbf{X}%
\mathbf{X}^{\top})(\mathbf{W}_{i}-\mathbf{T}_{i}^{k}\mathbf{S}_{i})^{\top}.$
(7)
Then, consider the Cholesky decomposition of $\mathbf{X}\mathbf{X}^{\top}$:
$\mathbf{X}\mathbf{X}^{\top}=\mathbf{L}\mathbf{L}^{\top},$
(8)
where $\mathbf{L}$ is a lower triangle matrix, meaning all its entries above the diagonal are zero.
REMARK 3.1.
If $\mathbf{X}\mathbf{X}^{\top}$ is not positive definite, which is rare but can occur in cases like the fc2 layer of OPT models, we can add $\lambda\mathbf{I}$ ($\lambda&gt;0$) to guarantee positive definiteness before Cholesky decomposition. Specifically, for any non-zero vector $\mathbf{v}$, $\mathbf{v}^{\top}(\mathbf{X}\mathbf{X}^{\top}+\lambda\mathbf{I})\mathbf{v}=\|%
\mathbf{X}^{\top}\mathbf{v}\|^{2}+\lambda\|\mathbf{v}\|^{2}&gt;0$.
By combining (7 [https://arxiv.org/html/2501.12956v2#S3.E7]) and (8 [https://arxiv.org/html/2501.12956v2#S3.E8]), we have:
$\displaystyle(\mathbf{W}_{i}-\mathbf{T}_{i}^{k}\mathbf{S}_{i})(\mathbf{X}%
\mathbf{X}^{\top})(\mathbf{W}_{i}-\mathbf{T}_{i}^{k}\mathbf{S}_{i})^{\top}$
(9)
$\displaystyle=$
$\displaystyle(\mathbf{W}_{i}-\mathbf{T}_{i}^{k}\mathbf{S}_{i})(\mathbf{L}%
\mathbf{L}^{\top})(\mathbf{W}_{i}-\mathbf{T}_{i}^{k}\mathbf{S}_{i})^{\top}$
(10)
$\displaystyle=$
$\displaystyle\|\mathbf{W}_{i}\mathbf{L}-\mathbf{T}_{i}^{k}\mathbf{S}_{i}%
\mathbf{L}\|^{2}.$
(11)
Leverage the structure of $\mathbf{L}$, we minimize (11 [https://arxiv.org/html/2501.12956v2#S3.E11]) using a back-substitution approach to efficiently derive a sub-optimal solution to (3 [https://arxiv.org/html/2501.12956v2#S3.E3]). Specifically, there is
$\displaystyle\|\mathbf{W}_{i}\mathbf{L}-\mathbf{T}_{i}^{k}\mathbf{S}_{i}%
\mathbf{L}\|^{2}$
(12)
$\displaystyle=$
$\displaystyle\sum_{j=0}^{n-1}\left(\left(\mathbf{W}_{i}\mathbf{L}\right)_{j}-%
\left(\mathbf{T}_{i}^{k}\mathbf{S}_{i}\mathbf{L}\right)_{j}\right)^{2}$
(13)
$\displaystyle=$
$\displaystyle\sum_{j=0}^{n-1}\!\left(\sum_{u=j}^{n-1}\left({\mathbf{W}_{i,u}}-%
\mathbf{T}_{i}^{k}\left(\mathbf{S}_{i}\right)_{:,u}\right)\mathbf{L}_{u,j}%
\right)^{2}.$
(14)
Following (14 [https://arxiv.org/html/2501.12956v2#S3.E14]), we can solve for $\mathbf{S}_{i}$ from the last column $(j=n-1)$ to the first column ($j=0$), minimizing each of the $n$ squared terms respectively. The $(n-1)$-th column of $\mathbf{L}$ has only one nonzero entry in rows $u\geq n-1$, namely $\mathbf{L}_{n-1,n-1}$. Therefore, for $j=n-1$, the residual involves a single term:
$\left(\mathbf{W}_{i,n-1}-\mathbf{T}_{i}^{k}\left(\mathbf{S}_{i}\right)_{:,n-1}%
\right)\mathbf{L}_{n-1,n-1}.$
(15)
Minimizing with respect to $(\mathbf{S}_{i})_{:,n-1}$ gives that it should select an element from $\mathbf{T}_{i}^{k}$ that satisfies
$\text{idx}=\operatorname*{argmin}_{s}\left|\mathbf{W}_{i,n-1}-\mathbf{T}^{k}_{%
i,s}\right|.$
(16)
Then, we set $(\mathbf{S}_{i})_{\text{idx},n-1}=1$ and all other elements in this column to 0.
Once $(\mathbf{S}_{i})_{:,n-1}$ is determined, the process moves to the $(n-2)$-th column. The residual becomes
$\displaystyle\left(\mathbf{W}_{i,n-2}-\mathbf{T}_{i}^{k}\left(\mathbf{S}_{i}%
\right)_{:,n-2}\right)\mathbf{L}_{n-2,n-2}$
(17)
$\displaystyle+$
$\displaystyle\left(\mathbf{W}_{i,n-1}-\mathbf{T}_{i}^{k}\left(\mathbf{S}_{i}%
\right)_{:,n-1}\right)\mathbf{L}_{n-1,n-2},$
(18)
where (18 [https://arxiv.org/html/2501.12956v2#S3.E18]) is a constant value given $\left(\mathbf{S}_{i}\right)_{:,n-1}$. In the following steps, we refer to $\mathbf{W}_{i,n-1}-\mathbf{T}_{i}^{k}\left(\mathbf{S}_{i}\right)_{:,n-1}$ as $r_{n-1}$. Then, we solve for $(\mathbf{S}_{i})_{:,n-2}$ by minimizing the square of (17 [https://arxiv.org/html/2501.12956v2#S3.E17])–(18 [https://arxiv.org/html/2501.12956v2#S3.E18]):
$\text{idx}=\operatorname*{argmin}_{s}\left|\mathbf{W}_{i,n-2}+\frac{r_{n-1}%
\mathbf{L}_{n-1,n-2}}{\mathbf{L}_{n-2,n-2}}-\mathbf{T}^{k}_{i,s}\right|,$
(19)
and we set $(\mathbf{S}_{i})_{\text{idx},n-2}=1$ and the rest of $(\mathbf{S}_{i})_{:,n-2}=0$.
This back-substitution process continues for $j=n-3,\dots,0$. At each step the element of $(\mathbf{S}_{i})_{\text{idx},j}$ set to 1 is determined as
$\text{idx}=\operatorname*{argmin}_{s}\left|\mathbf{W}_{i,j}+\frac{1}{\mathbf{L%
}_{j,j}}\sum_{u=j+1}^{n-1}r_{u}\mathbf{L}_{u,j}-\mathbf{T}^{k}_{i,s}\right|.$
(20)
where $r_{u}=\mathbf{W}_{i,u}-\mathbf{T}_{i}^{k}\left(\mathbf{S}_{i}\right)_{:,u}$.
Figure 2 [https://arxiv.org/html/2501.12956v2#S3.F2] illustrates the back-substitution framework for efficiently determining $\mathbf{S}_{i}$. Since the solution processes for $\mathbf{S}_{i},i=0,1,\dots,m-1$ are independent, similar to the batch solving of $\mathbf{T}_{i}$-subproblems described earlier, we can stack all $\mathbf{W}_{i}$ and $\mathbf{T}_{i}$ vectors row-wise and organize the $\mathbf{S}_{i}$ matrices into a tensor. This allows the back-substitution process to be performed for the entire problem using matrix operations, leveraging modern GPUs’ parallel processing capabilities to enhance overall efficiency.
Algorithm 1 GANQ: GPU-Adaptive Layer-Wise LUT-Based Non-Uniform Quantization
  Input: $\mathbf{W}\!\!\in\!\!\mathbb{R}^{m\times n}$, $\mathbf{X}\!\!\in\!\!\mathbb{R}^{n\times p}$, initial codebook $\mathbf{T}^{0}\!\!\in\!\!\mathbb{R}^{m\times 2^{N}}$, number of iterations $K$
  Output: Updated $\mathbf{T}^{K}$ and query matrix $\mathbf{Q}^{K}\!\!\in\!\!\{0,2^{N}\!-\!1\}^{m\times n}$
  Initialize $\mathbf{S}^{0}=\mathbf{0}^{m\times 2^{N}\times n}$ # tensor format
  Compute $\mathbf{H}=\mathbf{XX}^{\top}$
  Compute $\mathbf{L}=\text{Cholesky}(\mathbf{H})$ # Cholesky decomposition
  for $k\leftarrow 0$ to $K-1$ do
    Initialize $\mathbf{r}=\mathbf{0}^{m\times 1}$ # previous residual vector
    for $j\leftarrow n-1$ to $0$ do
      $\textbf{idx}=\operatorname*{argmin}_{\mathbf{s}}\left|\mathbf{W}_{:,j}+\frac{%
\mathbf{r}}{\mathbf{L}_{j,j}}-\mathbf{T}^{k}_{:,\mathbf{s}}\right|$# row-wise
      $\mathbf{Q}^{k+1}_{:,j}=\textbf{idx}$
      Update $\mathbf{S}^{k+1}_{:,:,j}$ using idx # one-hot encoding
      $\mathbf{r}=(\mathbf{W}_{:,j:}-\mathbf{T}^{k}\mathbf{S}_{:,:,j:}^{k+1})\mathbf{%
L}_{j:,j}$ # update residual
    end for
    $\mathbf{T}^{k+1}\!\!=\!\!\mathbf{W}\mathbf{H}(\mathbf{S}^{k+1})^{\top}\!((%
\mathbf{S}^{k+1})\mathbf{H}^{\top}\!(\mathbf{S}^{k+1})^{\top}\!)^{\dagger}$ # batch update
  end for
  Return $\mathbf{T}^{K},\mathbf{Q}^{K}$
Finally, the full pseudocode of GANQ for layer-wise LUT-based non-uniform quantization is presented in Algorithm 1 [https://arxiv.org/html/2501.12956v2#alg1].
