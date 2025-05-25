import numpy as np
import pyopencl as cl

def setup_opencl():
    # Mendapatkan platform dan device OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    
    # Menampilkan informasi GPU yang digunakan
    print(f"Platform: {platform.name}")
    print(f"Device: {device.name}")
    
    # Membuat context dan queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    return platform, device, context, queue

def load_kernel(context):
    # Membaca kernel dari file
    with open("kernel.cl", "r") as f:
        kernel_src = f.read()
    
    return kernel_src

def generate_points(num_points, x_min, x_max, y_min, y_max):
    # Membuat titik-titik acak dalam rentang yang ditentukan
    points = np.array([(np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)) for _ in range(num_points)], dtype=np.int32)
    return points

def visualize_results(points, triangles):
    import matplotlib.pyplot as plt
    # Visualisasi titik-titik dan segitiga siku-siku yang ditemukan
    for triangle in triangles:
        plt.plot([triangle[0][0], triangle[1][0]], [triangle[0][1], triangle[1][1]], 'r-')
        plt.plot([triangle[1][0], triangle[2][0]], [triangle[1][1], triangle[2][1]], 'r-')
        plt.plot([triangle[2][0], triangle[0][0]], [triangle[2][1], triangle[0][1]], 'r-')
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()
