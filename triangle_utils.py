import numpy as np

def is_right_angle(v1, v2, v3, tolerance=5):
    # Menghitung panjang sisi segitiga
    a = np.linalg.norm(v1 - v2)
    b = np.linalg.norm(v2 - v3)
    c = np.linalg.norm(v3 - v1)
    
    # Menggunakan aturan Pythagoras untuk mendeteksi sudut siku-siku
    sides = sorted([a, b, c])
    # Periksa apakah sisi terbesar kuadratnya hampir sama dengan jumlah kuadrat sisi lainnya
    return np.abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < tolerance

def find_right_triangles(points):
    triangles = []
    used_points = set()  # Titik yang sudah digunakan dalam segitiga
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            for k in range(j+1, len(points)):
                # Mengambil tiga titik
                p1, p2, p3 = points[i], points[j], points[k]
                
                # Menghindari penggunaan titik yang sama lebih dari satu kali
                if i in used_points or j in used_points or k in used_points:
                    continue
                
                # Menghitung apakah ketiga titik membentuk segitiga siku-siku
                if is_right_angle(p1, p2, p3):
                    triangles.append([p1, p2, p3])
                    used_points.update([i, j, k])  # Tandai titik yang digunakan
    return triangles
