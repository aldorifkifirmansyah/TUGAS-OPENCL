__kernel void count_right_triangles(__global const int2* points,
                                    const int num_points,
                                    __global int* triangle_count,
                                    __global int* triangle_indices,
                                    __global int* used_points) {
    int i = get_global_id(0);
    if (i >= num_points - 2) return;

    for (int j = i + 1; j < num_points - 1; ++j) {
        for (int k = j + 1; k < num_points; ++k) {
            // Cek apakah titik sudah digunakan
            if (atomic_or(&used_points[i], 0) || atomic_or(&used_points[j], 0) || atomic_or(&used_points[k], 0))
                continue;

            int2 p1 = points[i];
            int2 p2 = points[j];
            int2 p3 = points[k];

            float a2 = (float)((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
            float b2 = (float)((p2.x - p3.x)*(p2.x - p3.x) + (p2.y - p3.y)*(p2.y - p3.y));
            float c2 = (float)((p3.x - p1.x)*(p3.x - p1.x) + (p3.y - p1.y)*(p3.y - p1.y));

            float max_side = fmax(fmax(a2, b2), c2);
            float sum_other = a2 + b2 + c2 - max_side;

            if (fabs(sum_other - max_side) < 5.0f) {
                // Tandai titik sebagai sudah digunakan (atomic agar aman)
                if (atomic_xchg(&used_points[i], 1) == 0 &&
                    atomic_xchg(&used_points[j], 1) == 0 &&
                    atomic_xchg(&used_points[k], 1) == 0) {
                    int idx = atomic_inc(&triangle_count[0]);
                    int base = idx * 3;
                    triangle_indices[base]     = i;
                    triangle_indices[base + 1] = j;
                    triangle_indices[base + 2] = k;
                }
            }
        }
    }
}
