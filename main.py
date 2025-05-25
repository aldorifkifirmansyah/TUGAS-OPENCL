import numpy as np
import pyopencl as cl
import time
import matplotlib.pyplot as plt
from opencl_utils import setup_opencl, load_kernel, generate_points, visualize_results
from triangle_utils import find_right_triangles

def main():
    # Set jumlah titik acak
    num_points = 1000
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

    # Menjalankan kernel untuk mendeteksi segitiga siku-siku
    program = cl.Program(context, kernel).build()
    program.check_right_triangles(queue, (len(points),), None, points_buffer, np.int32(len(points)), results_buffer)

    # Menyalin hasil dari buffer
    cl.enqueue_copy(queue, results, results_buffer).wait()

    # Menunggu eksekusi OpenCL selesai
    queue.finish()

    # Selesai pengukuran waktu OpenCL
    end_time = time.time()
    execution_time_opencl = end_time - start_time
    print(f"Waktu eksekusi OpenCL: {execution_time_opencl:.6f} detik")

    # Menemukan segitiga siku-siku dengan memastikan eksklusivitas titik
    triangles = find_right_triangles(points)

    # Visualisasi hasil akhir (state akhir)
    print("Visualisasi segitiga siku-siku yang ditemukan...")
    visualize_results(points, triangles)  # Visualisasi titik dengan segitiga

if __name__ == "__main__":
    main()
