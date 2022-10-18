{
  ## for windows, be sure to do node-gyp rebuild -msvs_version=2013, 
  ## and run under a msvs shell

  ## for all targets 
  'conditions': [
    [ 'OS=="win"', {'variables': {'obj': 'obj'}}, 
    {'variables': {'obj': 'o'}}]],

  "targets": [
{
 "target_name": "cuno",
 "sources": [ 
    "./src/main.cu",
    "./src/kernel/kernels/dot.cu",
    "./src/bindings/bindings/DotProd.cu",
    "./src/utils/utils.cu",
    "./src/dannjs/*.cu"
  ], 

 'rules': [{
     'extension': 'cu',           
     'inputs': ['<(RULE_INPUT_PATH)'],
     'outputs':[ '<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).<(obj)'],
     'conditions': [
      [ 'OS=="win"',  
        {'rule_name': 'cuda on windows',
         'message': "compile cuda file on windows",
         'process_outputs_as_sources': 0,
         'action': ['nvcc -c <(_inputs) -o  <(_outputs)'],
         }, 
       {'rule_name': 'cuda on linux',
         'message': "compile cuda file on linux",
         'process_outputs_as_sources': 1,
         'action': ['nvcc','-Xcompiler','-fpic','-c',
            '<@(_inputs)','-o','<@(_outputs)'],
    }]]}],

   'conditions': [
    [ 'OS=="mac"', {
      'libraries': ['-framework CUDA'],
      'include_dirs': ['/usr/local/include'],
      'library_dirs': ['/usr/local/lib'],
    }],
    [ 'OS=="linux"', {
      'libraries': ['-lcuda', '-lcudart'],
      'include_dirs': ['/usr/local/include', '/opt/cuda/include'],
      'library_dirs': ['/usr/local/lib', '/opt/cuda/lib64','/usr/local/cuda/lib64'],
    }],
    [ 'OS=="win"', {
      'conditions': [
        ['target_arch=="x64"',
          {
            'variables': { 'arch': 'x64' }
          }, {
            'variables': { 'arch': 'Win32' }
          }
        ],
      ],
      'variables': {
        'cuda_root%': '$(CUDA_PATH)'
      },
      'libraries': [
        '-l<(cuda_root)/lib/<(arch)/cuda.lib',
        '-l<(cuda_root)/lib/<(arch)/cudart.lib',
      ],
      "include_dirs": [
        "<(cuda_root)/include",
      ],
    }, {
      "include_dirs": [
        "/usr/local/cuda/include"
      ],
    }]
  ]
}
]
}
