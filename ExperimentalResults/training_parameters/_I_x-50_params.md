# Exp: I_x-50
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
discriminator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f5d4c4ed400>
discriminator_metrics, ['accuracy']
generator_loss, {'possibility': 'binary_crossentropy', 'I_x': 'mean_squared_error', 'I_y': 'mean_squared_error', 'I_r': 'mean_squared_error'}
generator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f5d5ea514e0>
generator_metrics, ['accuracy', 'mean_squared_error', <__main__.MetricsAverageSecondMomentOfArea object at 0x7f5d42de4438>]
generator_loss_weights, {'possibility': 1.0, 'I_x': 50.0, 'I_y': 0.0, 'I_r': 0.0}
EXPERIMENTAL_NAME, I_x-50
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
FIXED_NOISE_FOR_PREDICT, [[ 0.18866129  0.2030965  -1.0723804  ...  0.83636811 -1.23467649
   1.84398292]
 [ 0.99636116  0.43555346 -1.84714244 ... -0.71062067 -0.14645201
  -0.19332728]
 [ 1.69299543  0.0175794  -1.13668296 ... -1.5825674  -1.53055879
   0.62457807]
 ...
 [-0.54100914  1.63545806  0.94616429 ...  0.73560186 -0.67016753
  -1.12108629]
 [ 1.60022512 -1.22374189 -0.32521664 ... -1.46829287  0.44243925
   0.97696089]
 [-0.15701791 -0.02646659  0.49167956 ...  0.77865621  1.05217511
   0.26898918]]
opt_I_x, [[0.]
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
opt_I_r, [[1.]
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
