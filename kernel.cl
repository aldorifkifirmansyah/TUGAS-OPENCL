__kernel void check_right_triangles(__global const int2* points, const int num_points, __global int* results) {
    int idx = get_global_id(0);
    
    if (idx < num_points) {
        // Memeriksa setiap triplet titik
        for (int j = idx + 1; j < num_points; j++) {
            for (int k = j + 1; k < num_points; k++) {
                // Titik A, B, C
                int2 A = points[idx];
                int2 B = points[j];
                int2 C = points[k];
                
                // Vektor AB, BC, dan AC
                int2 AB = (int2)(B.x - A.x, B.y - A.y);
                int2 BC = (int2)(C.x - B.x, C.y - B.y);
                int2 AC = (int2)(C.x - A.x, C.y - A.y);
                
                // Dot product untuk memeriksa sudut 90 derajat
                int dot_AB_AC = AB.x * AC.x + AB.y * AC.y;
                int dot_AB_BC = AB.x * BC.x + AB.y * BC.y;
                int dot_AC_BC = AC.x * BC.x + AC.y * BC.y;
                
                // Toleransi error untuk mendekati nol
                int tolerance = 1e-6;
                
                if (abs(dot_AB_AC) < tolerance || abs(dot_AB_BC) < tolerance || abs(dot_AC_BC) < tolerance) {
                    results[idx] = 1;  // Menandai hasil segitiga siku-siku
                }
            }
        }
    }
}
