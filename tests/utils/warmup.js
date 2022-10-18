function warmup(func) {
    let N = 10;
    func(
        new Array(N * N).fill(1),
        new Array(N * N).fill(1),
        N, N, N
    )
}

module.exports = { warmup };