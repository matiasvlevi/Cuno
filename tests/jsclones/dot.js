function dot(a, b, M, N, P) {
    let c = new Array(M, P).fill(0);
    for (let row = 0; row < M; row++){
        for (let col = 0; col < P; col++){
            let sum = 0;
            for (let k = 0; k < N; k++) {
                sum += a[row * N + k] * b[k * P + col];
            }
            c[row * P + col] = sum;
        }
    }
    return c;
}

module.exports = { dot };