# Exp: I_r-500
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
discriminator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f6c0de81550>
discriminator_metrics, ['accuracy']
generator_loss, {'possibility': 'binary_crossentropy', 'I_x': 'mean_squared_error', 'I_y': 'mean_squared_error', 'I_r': 'mean_squared_error'}
generator_optimizer, <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f6bff8aa9e8>
generator_metrics, ['accuracy', 'mean_squared_error', <__main__.MetricsAverageSecondMomentOfArea object at 0x7f6c064683c8>]
generator_loss_weights, {'possibility': 1.0, 'I_x': 0.0, 'I_y': 0.0, 'I_r': 500.0}
EXPERIMENTAL_NAME, I_r-500
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
FIXED_NOISE_FOR_PREDICT, [[-0.02363839  0.27166974  1.77975822 ...  0.25961941  0.71001624
  -1.29421263]
 [-0.8634511  -0.25835121 -0.93102609 ... -2.32456918  1.17216895
   0.24615207]
 [ 0.3138134  -0.27851782 -0.73990218 ... -0.10449399  0.63468011
  -1.03310636]
 ...
 [ 1.02203151 -0.08833952  0.88268426 ...  1.5062615   0.41130043
   1.17639552]
 [ 1.06082979 -0.36522396  0.6490142  ... -0.04753034  1.79309532
   1.71811599]
 [-0.62279206  0.74623928 -0.52734202 ... -0.10469371 -0.20157399
  -0.18185206]]
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