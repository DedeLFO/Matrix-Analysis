"# Matrix-Analysis" 
from mpi4py import MPI
import numpy as np
import time

# Configurações de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Identificação do processo
size = comm.Get_size()   # Número total de processos

# Função de multiplicação de submatrizes
def matrix_multiply(A, B):
    return np.dot(A, B)

# Teste para diferentes tamanhos de matriz
matrix_sizes = [100, 500, 1000]  # Tamanhos de matriz para teste
results = []

for N in matrix_sizes:
    # Inicialização das matrizes apenas no processo mestre
    if rank == 0:
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        C = np.zeros((N, N))  # Matriz resultado
        
        # Dividir a matriz A em "fatias" para os processos
        rows_per_proc = N // size
        rows_extra = N % size
        offset = 0
        
        # Início da medição de tempo
        start_time = time.time()
    else:
        A = None
        B = None
        C = None
        rows_per_proc = None

    # Envio do número de linhas por processo a todos os processos
    rows_per_proc = comm.bcast(rows_per_proc if rank == 0 else None, root=0)

    # Cada processo aloca o espaço necessário para receber suas linhas de A
    local_rows = rows_per_proc + (1 if rank < N % size else 0)
    local_A = np.zeros((local_rows, N))

    # O processo mestre envia as "fatias" de A e toda a matriz B para cada processo
    if rank == 0:
        for i in range(1, size):
            start_row = offset
            end_row = offset + rows_per_proc + (1 if i < N % size else 0)
            comm.Send([A[start_row:end_row, :], MPI.DOUBLE], dest=i, tag=77)
            offset = end_row
        local_A = A[:local_rows, :]
    else:
        B = np.empty((N, N))
        comm.Recv(local_A, source=0, tag=77)

    # Broadcast da matriz B para todos os processos
    comm.Bcast(B, root=0)

    # Multiplicação de submatrizes local
    local_C = matrix_multiply(local_A, B)

    # Reunindo a matriz C completa no processo mestre
    if rank == 0:
        C[:local_C.shape[0], :] = local_C
        offset = local_C.shape[0]
        for i in range(1, size):
            start_row = offset
            end_row = start_row + rows_per_proc + (1 if i < N % size else 0)
            comm.Recv(C[start_row:end_row, :], source=i, tag=88)
            offset = end_row
    else:
        comm.Send(local_C, dest=0, tag=88)

    # Final da medição de tempo e armazenamento dos resultados
    if rank == 0:
        end_time = time.time()
        exec_time = end_time - start_time
        results.append((N, exec_time))
        print(f"Tamanho da matriz: {N}x{N} | Tempo de execução: {exec_time:.2f} segundos")

# Apenas o processo mestre imprime os resultados finais de todas as execuções
if rank == 0:
    print("\nResultados finais de execução para diferentes tamanhos de matriz:")
    for result in results:
        print(f"Matriz {result[0]}x{result[0]} - Tempo: {result[1]:.2f} segundos")
