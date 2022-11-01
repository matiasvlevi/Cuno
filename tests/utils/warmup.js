function warmup(func) {
    let N = 128;
    func(
      new Array(N).fill(new Array(N).fill(1)),
      new Array(N).fill(new Array(N).fill(1))
    )
}

module.exports = { warmup };