# Exp: I_r+75
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
discriminator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f6c1daa47f0>
discriminator_metrics, ['accuracy']
generator_loss, {'possibility': 'binary_crossentropy', 'I_x': 'mean_squared_error', 'I_y': 'mean_squared_error', 'I_r': 'mean_squared_error'}
generator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f6c154a9978>
generator_metrics, ['accuracy', 'mean_squared_error', <__main__.MetricsAverageSecondMomentOfArea object at 0x7f6c18afe710>]
generator_loss_weights, {'possibility': 1.0, 'I_x': 0.0, 'I_y': 0.0, 'I_r': 75.0}
EXPERIMENTAL_NAME, I_r+75
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
FIXED_NOISE_FOR_PREDICT, [[ 1.12973874 -0.8618458  -0.1868789  ... -0.65717311  0.33564419
   0.02504086]
 [ 1.7570755   0.05828725 -0.3790879  ...  0.21242209  1.3141633
  -1.63179841]
 [-0.67483092 -0.56347716  0.15275025 ...  0.01652471  1.3239777
   0.572948  ]
 ...
 [ 1.86069636 -1.09390019 -0.06314891 ... -2.66587795  0.13277333
  -0.06833068]
 [ 2.36511055  0.40020694  1.54917256 ... -0.86023681 -2.20261991
   0.5850942 ]
 [-0.68756277  0.50059897 -0.47942387 ... -0.69049655 -1.35393161
  -0.20897172]]
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
