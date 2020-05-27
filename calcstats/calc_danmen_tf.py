import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

"""
ピクセルの考え方
↓
0.0 0.5 1.0 1.5 2.0 ....   27.0 27.5 28.0 ←座標
 | ■ | ■ | ■ | □ |  ....  □ |  ■ | ■ | ← ピクセル値(「|」はピクセルとピクセルの隙間)
"""

class SecondMomentOfArea:
    def __init__(self, img_shape):
        """
        1. ピクセルの中心が重心, 0.5^27.5
        2. (全て1の場合) 画像の中心(14, 14)が重心
        となるようにする
        """
        # [0, 1, 2, ..., 27]
        arange_x = 0.5 + np.arange(0, img_shape[0], 1) # 縦のピクセル数
        arange_y = 0.5 + np.arange(0, img_shape[1], 1) # 横のピクセル数

        # ピクセルの中心が重心, 0.5^27.5
        distance_vector_x = np.asarray([0.5+d for d in range(img_shape[1])])
        distance_matrix_x = np.tile(distance_vector_x, (img_shape[0], 1)) # xからの距離 (MNISTの端 → 27.5)
        distance_matrix_y = distance_matrix_x.T # yからの距離

        """
        正規化用マトリックス(この設定における最大 = 全ピクセル使用)
        """
        # 縦方向(y)に対する断面二次モーメント(I_x)を正規化するため、最大の断面二次モーメントを求める
        matrix_for_norm_I_x = np.tile(np.abs(arange_y - img_shape[0]/2.0), (img_shape[1], 1)).T
        norm_I_x = np.sum(matrix_for_norm_I_x)

        # 横方向(x), I_y
        matrix_for_norm_I_y = np.tile(np.abs(arange_x - img_shape[1]/2.0), (img_shape[0], 1)).T
        norm_I_y = np.sum(matrix_for_norm_I_y)

        # to TFconstant
        self.arange_x = tf.constant(arange_x, dtype=tf.float32) # (28, )
        self.arange_y = tf.constant(arange_y, dtype=tf.float32) # (28,)
        self.distance_matrix_x = tf.constant(distance_matrix_x[np.newaxis, :, :, np.newaxis], dtype=tf.float32) # (1, 28, 28, 1)
        self.distance_matrix_y = tf.constant(distance_matrix_y[np.newaxis, :, :, np.newaxis], dtype=tf.float32) #(1, 28, 28, 1)
        self.norm_I_x = tf.constant(norm_I_x, dtype=tf.float32) #()
        self.norm_I_y = tf.constant(norm_I_y, dtype=tf.float32) #()

        plt.imshow(self.distance_matrix_x[0, :, :, 0]/28.0, cmap = "gray", vmin=0.0, vmax=1.0)
        plt.show()

        plt.imshow(self.distance_matrix_y[0, :, :, 0]/28.0, cmap = "gray", vmin=0.0, vmax=1.0)
        plt.show()

    @tf.function
    def calc_smoa(self, img):
        """
        断面二次モーメントの計算
        """

        """
        中立軸の計算
        """
        # 密度。ゼロじゃない画素の割合　
        density = (tf.reduce_sum(img, axis=[1, 2], keepdims=True)/(img.shape[1]*img.shape[2]))
        # (1, 28, 28, 1) x (None, 28, 28, 1) -> (None, 28, 28, 1)
        x_moment = tf.math.divide_no_nan(
            tf.math.multiply(self.distance_matrix_x, img), density) # ゼロ除算回避付
        y_moment = tf.math.divide_no_nan(
            tf.math.multiply(self.distance_matrix_y, img), density)

        # (None, 28, 28, 1) -> (None, )
        neutral_axis_x = tf.math.reduce_mean(x_moment, axis=[1, 2])
        neutral_axis_y = tf.math.reduce_mean(y_moment, axis=[1, 2])

        """
        断面二次モーメント (縦)
        I_x = ∫_A y^2 dA
        """
        # sub: (None, 28, ) - (None, ) -> abs: (None, 28)
        dy = tf.math.abs(self.arange_y - neutral_axis_y)
        # (None, 28) -> (None, 1, 28)
        dy = tf.reshape(dy, shape=[-1, img.shape[1], 1])
        # (None, 1, 28) -> (None, 28, 28)
        matrix_x = tf.tile(dy, multiples=[1, 1, img.shape[2]])
        # (None, 28, 28) -> (None, 28, 28, 1)
        matrix_x = tf.expand_dims(matrix_x, 3)
        # (None, 28, 28, 1)x(None, 28, 28, 1) -> (None, 28, 28, 1) -> (None,)
        I_x = tf.math.reduce_sum(tf.math.multiply(matrix_x, img), axis=[1, 2])/self.norm_I_x

        """
        断面二次モーメント (横)
        I_y = ∫_A x^2 dA
        """
        # sub: (None, 28, ) - (None, ) -> abs: (None, 28)
        dx = tf.math.abs(self.arange_x - neutral_axis_x)
        # (None, 28) -> (None, 28, 1)
        dx = tf.reshape(dx, shape=[-1, 1, img.shape[2]])
        # (None, 1, 28) -> (None, 28, 28)
        matrix_y = tf.tile(dx, multiples=[1, img.shape[1], 1])
        # (None, 28, 28) -> (None, 28, 28, 1)
        matrix_y = tf.expand_dims(matrix_y, 3)
        # (None, 28, 28, 1)x(None, 28, 28, 1) -> (None, 28, 28, 1) -> (None,)
        I_y = tf.math.reduce_sum(tf.math.multiply(matrix_y, img), axis=[1, 2])/self.norm_I_y
        """
        断面二次極モーメント (正規化のため 2.0 で割る)
        """
        I_r = (I_x + I_y)/2.0

        return I_x, I_y, I_r

def calcMoIoA():
    smoa = SecondMomentOfArea(img_shape = (28, 28))

    """
    線画薄い場所＝面積が減っている とする。
    実際のプリントとは異なってしまうが、今回は仕方ない。
    """
    (train_X, train_y), (_, _) = tf.keras.datasets.mnist.load_data()
    train_X = train_X.astype(np.float64).reshape((train_X.shape[0], 28, 28))/255.0

    """
    img = np.zeros(shape=[28, 28])
    img[0:14, 0:28] = 1
    img_tensor = tf.constant(img[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    print(img_tensor.dtype)
    """

    I_x_holder = [[] for i in range(10)]
    I_y_holder = [[] for i in range(10)]
    I_r_holder = [[] for i in range(10)]

    print(f"| Number | I_x mean(var) | I_y mean(var) | I_r mean(var) |")
    for number in range(10):

        number_arr = train_X[train_y == number] # 0, 1, 2, ..., 9

        start_time = time.time()
        for i in range(number_arr.shape[0]):

            img_tensor = tf.constant(number_arr[i][np.newaxis, :, :, np.newaxis], dtype=tf.float32)
            I_x, I_y, I_r = smoa.calc_smoa(img_tensor)

            I_x_holder[number].append(I_x[0, 0].numpy())
            I_y_holder[number].append(I_y[0, 0].numpy())
            I_r_holder[number].append(I_r[0, 0].numpy())

        #print(number)
        #plt.imshow(img, cmap = "gray", vmin=0.0, vmax=1.0)
        #plt.show()

        print(f"| {number} | {np.mean(I_x_holder[number]):.3f}({np.std(I_x_holder[number]):.3f}) | {np.mean(I_y_holder[number]):.3f}({np.std(I_y_holder[number]):.3f}) | {np.mean(I_r_holder[number]):.3f}({np.std(I_r_holder[number]):.3f}) |")

    """
    calc stats
    """
    all_I_x = np.concatenate([np.asarray(I_x_holder[i]) for i in range(10)], axis=0) # 各数字ごとにデータ数が違うのでまとめて計算する
    all_I_y = np.concatenate([np.asarray(I_y_holder[i]) for i in range(10)], axis=0) # 各数字ごとにデータ数が違うのでまとめて計算する
    all_I_r = np.concatenate([np.asarray(I_r_holder[i]) for i in range(10)], axis=0) # 各数字ごとにデータ数が違うのでまとめて計算する
    global_mean_I_x, global_std_I_x = all_I_x.mean(), all_I_x.std()
    global_mean_I_y, global_std_I_y = all_I_y.mean(), all_I_y.std()
    global_mean_I_r, global_std_I_r = all_I_r.mean(), all_I_r.std()

    print("(global)")
    print("| I_x(std) | I_y(std) | I_r(std)|")
    print(f"| {global_mean_I_x:.3f}({global_std_I_x:.3f}) | {global_mean_I_y:.3f}({global_std_I_y:.3f}) | {global_mean_I_r:.3f}({global_std_I_r:.3f})  |")

    """
    boxplot
    """

    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gray')

    labels = [str(i) for i in range(10)]

    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    ax.boxplot([I_x_holder[i] for i in range(10)], labels=labels, whis="range")
    ax.set_xlabel('Number')
    ax.set_ylabel('I_x')
    ax.set_ylim(0, 0.3)

    ax = fig.add_subplot(3, 1, 2)
    ax.boxplot([I_y_holder[i] for i in range(10)], labels=labels, whis="range")
    ax.set_xlabel('Number')
    ax.set_ylabel('I_y')
    ax.set_ylim(0, 0.3)

    ax = fig.add_subplot(3, 1, 3)
    ax.boxplot([I_r_holder[i] for i in range(10)], labels=labels, whis="range")
    ax.set_xlabel('Number')
    ax.set_ylabel('I_r')
    ax.set_ylim(0, 0.3)
    plt.show()

calcMoIoA()
