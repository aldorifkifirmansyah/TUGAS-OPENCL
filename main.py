import numpy as np
import pyopencl as cl
import time
import matplotlib.pyplot as plt
from opencl_utils import setup_opencl, load_kernel, generate_points, visualize_results


def main():
    # Set jumlah titik acak
    num_points = 250
    points = generate_points(num_points, 0, 500, 0, 500)

    # Menyiapkan OpenCL (platform, device, context, queue)
    platform, device, context, queue = setup_opencl()

    # Memuat OpenCL kernel dari file
    kernel = load_kernel(context)

    # Membuat buffer untuk data titik
    points_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=points)

    # Membuat buffer untuk hasil
    results = np.zeros(len(points), dtype=np.int32)
    results_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, results.nbytes)

    # Menampilkan visualisasi titik acak (state awal)
    print("Visualisasi titik acak...")
    visualize_results(points, [])  # Visualisasi titik awal tanpa segitiga

    # Mulai pengukuran waktu untuk eksekusi OpenCL
    start_time = time.time()

    # Alokasikan buffer untuk jumlah segitiga
    triangle_count = np.zeros(1, dtype=np.int32)
    triangle_count_buffer = cl.Buffer(
    context,
    cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
    hostbuf=triangle_count
)

    max_triangles = 10000  # kapasitas maksimum segitiga (bisa disesuaikan)
    triangle_indices = np.zeros(max_triangles * 3, dtype=np.int32)
    triangle_indices_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, triangle_indices.nbytes)

    # Menjalankan kernel untuk mendeteksi segitiga siku-siku
    program = cl.Program(context, kernel).build()
    
    used_points = np.zeros(num_points, dtype=np.int32)
    used_points_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=used_points)

    program.count_right_triangles(queue, (num_points,), None,
    points_buffer,
    np.int32(num_points),
    triangle_count_buffer,
    triangle_indices_buffer,
    used_points_buffer)

    cl.enqueue_copy(queue, triangle_count, triangle_count_buffer).wait()
    cl.enqueue_copy(queue, triangle_indices, triangle_indices_buffer).wait()

    # Ambil hanya jumlah yang valid
    triangle_indices = triangle_indices[:triangle_count[0] * 3].reshape((-1, 3))

    # Ubah ke koordinat titik nyata
    triangles = [[points[i], points[j], points[k]] for i, j, k in triangle_indices]

    print(f"Jumlah segitiga siku-siku (OpenCL): {triangle_count[0]}")



    # Menyalin hasil dari buffer
    cl.enqueue_copy(queue, results, results_buffer).wait()

    # Menunggu eksekusi OpenCL selesai
    queue.finish()

    # Selesai pengukuran waktu OpenCL
    end_time = time.time()
    execution_time_opencl = end_time - start_time
    print(f"Waktu eksekusi OpenCL: {execution_time_opencl:.6f} detik")

    # Visualisasi hasil akhir (state akhir)
    print("Visualisasi segitiga siku-siku yang ditemukan...")
    visualize_results(points, triangles)  # Visualisasi titik dengan segitiga

if __name__ == "__main__":
    main()