# Exp: vanillaGAN
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
discriminator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f5fea2a9b00>
discriminator_metrics, ['accuracy']
generator_loss, {'possibility': 'binary_crossentropy', 'I_x': 'mean_squared_error', 'I_y': 'mean_squared_error', 'I_r': 'mean_squared_error'}
generator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f5fea2a9b38>
generator_metrics, ['accuracy', 'mean_squared_error', <__main__.MetricsAverageSecondMomentOfArea object at 0x7f5fea2a9b70>]
generator_loss_weights, {'possibility': 1.0, 'I_x': 0.0, 'I_y': 0.0, 'I_r': 0.0}
EXPERIMENTAL_NAME, vanillaGAN
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
FIXED_NOISE_FOR_PREDICT, [[ 0.5345252   1.3868703  -0.71133475 ...  0.48493405  0.89279147
   1.10154587]
 [ 0.45799111 -0.54890714  0.40812428 ...  0.00610612  0.8358637
   0.6906576 ]
 [ 0.30838709 -1.19869506 -0.22800299 ... -0.88434073  0.0462956
  -1.29964995]
 ...
 [ 0.90602106  1.16977014 -0.48948652 ...  0.16575517 -1.13777917
   0.40643064]
 [ 0.01343677 -1.87553217 -0.23404244 ...  1.67969978 -0.59199077
  -0.78137374]
 [-0.48484404 -1.29065232  0.80157884 ...  0.92704039 -1.02652435
  -0.4556671 ]]
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
