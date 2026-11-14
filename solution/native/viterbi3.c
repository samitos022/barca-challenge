/* Viterbi over three states (0 home, 1 away, 2 dead) with transition costs that may change frame by
 * frame. lp: n x 3 log-emissions; ct: cost of a home <-> away change entering frame t; cin: alive ->
 * dead entering t; cout: dead -> alive entering t. Writes the path into out (int8). Plain C, no
 * dependencies: built by solution/viterbi.py with the system compiler. */
#include <stdlib.h>
#include <stdint.h>
void viterbi3(const double *lp, const double *ct, const double *cin, const double *cout, long n, int8_t *out) {
    if (n <= 0) return;
    int8_t *back = (int8_t *)malloc((size_t)n * 3);
    double s0 = lp[0], s1 = lp[1], s2 = lp[2];
    for (long t = 1; t < n; t++) {
        const double *e = lp + 3 * t;
        double c = ct[t], i = cin[t], o = cout[t];
        double n0, n1, n2; int8_t b0, b1, b2;
        /* to home */
        n0 = s0; b0 = 0;
        if (s1 - c > n0) { n0 = s1 - c; b0 = 1; }
        if (s2 - o > n0) { n0 = s2 - o; b0 = 2; }
        /* to away */
        n1 = s1; b1 = 1;
        if (s0 - c > n1) { n1 = s0 - c; b1 = 0; }
        if (s2 - o > n1) { n1 = s2 - o; b1 = 2; }
        /* to dead */
        n2 = s2; b2 = 2;
        if (s0 - i > n2) { n2 = s0 - i; b2 = 0; }
        if (s1 - i > n2) { n2 = s1 - i; b2 = 1; }
        s0 = n0 + e[0]; s1 = n1 + e[1]; s2 = n2 + e[2];
        back[3 * t] = b0; back[3 * t + 1] = b1; back[3 * t + 2] = b2;
    }
    int8_t k = 0; double best = s0;
    if (s1 > best) { best = s1; k = 1; }
    if (s2 > best) { best = s2; k = 2; }
    for (long t = n - 1; t >= 0; t--) {
        out[t] = k;
        if (t > 0) k = back[3 * t + k];
    }
    free(back);
}
