# Exp: I_r-25
## params
tf_version, 2.2.0
sys_versionm, 3.6.9 (default, Apr 18 2020, 01:56:04) 
[GCC 8.4.0]
ITERATION_SPAN, 4000
NUM_EPOCHS, 20
BATCH_SIZE, 50
MAX_ITERATION, 1200
NOISE_DIM, 128
discriminator_loss, binary_crossentropy
discriminator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7fa0b26905f8>
discriminator_metrics, ['accuracy']
generator_loss, {'possibility': 'binary_crossentropy', 'I_x': 'mean_squared_error', 'I_y': 'mean_squared_error', 'I_r': 'mean_squared_error'}
generator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7fa0b2690828>
generator_metrics, ['accuracy', 'mean_squared_error', <__main__.MetricsAverageSecondMomentOfArea object at 0x7fa0b2690588>]
generator_loss_weights, {'possibility': 1.0, 'I_x': 0.1, 'I_y': 0.0, 'I_r': 25.0}
EXPERIMENTAL_NAME, I_r-25
REAL, [[1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]]
FAKE, [[0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]]
FIXED_NOISE_FOR_PREDICT, [[-0.73691522 -1.58979222  0.45599964 ... -0.35885409 -0.81971875
   0.19711753]
 [ 0.51766185  0.39929686 -0.53605029 ...  0.80092005 -0.50356178
  -0.7688571 ]
 [ 0.88717908 -0.33887124 -0.52180128 ...  1.96879111  1.78852805
   0.20472357]
 ...
 [-0.52772801  1.19241714  0.91854959 ...  0.11437969  0.75637607
  -0.88021336]
 [ 0.95307447 -0.530968    1.82783919 ... -1.17554001 -0.17156011
  -0.29230288]
 [ 0.8387544   0.675759   -0.0902481  ... -0.11932529  1.21562926
   1.68040354]]
opt_I_x, [[1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]]
opt_I_y, [[1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]]
opt_I_r, [[0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]]
