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
    """
    断面二次モーメントに関するクラス
    """
    def __init__(self):
        self.img_shape = (28, 28) #[y, x]

        """
        1. ピクセルの中心が重心, 0.5^27.5
        2. (全て1の場合) 画像の中心(14, 14)が重心
        となるようにする
        """
        self.arange_y = 0.5 + np.arange(0, self.img_shape[0], 1)# [0, 1, 2, ..., 27]
        self.arange_x = 0.5 + np.arange(0, self.img_shape[1], 1)# [0, 1, 2, ..., 27]
        distance_vector_x =  self.arange_x

        self.distance_matrix_x = np.tile(distance_vector_x, (self.img_shape[0], 1)) # xからの距離 (MNISTの端 → 27.5)
        self.distance_matrix_y = self.distance_matrix_x.T # yからの距離

        """
        正規化用マトリックス(この設定における最大 = 全ピクセル使用)
        """
        # 縦方向(y)に対する断面二次モーメント(I_x)を正規化するため、最大の断面二次モーメントを求める
        matrix_for_norm_I_x = np.tile(np.abs(self.arange_y - self.img_shape[0]/2.0), (self.img_shape[1], 1)).T
        self.norm_I_x = np.sum(matrix_for_norm_I_x)

        # 横方向(x), I_y
        matrix_for_norm_I_y = np.tile(np.abs(self.arange_x - self.img_shape[1]/2.0), (self.img_shape[0], 1)).T
        self.norm_I_y = np.sum(matrix_for_norm_I_y)

    def calc_smoa(self, img):

        density = (np.sum(img)/784) # 密度。ゼロじゃない画素の割合　
        #print("density", density)

        neutral_axis_x = np.mean(self.distance_matrix_x*(img))/density # x=0から測ったxの重心の位置(range:[0.0-1.0])。
        neutral_axis_y = np.mean(self.distance_matrix_y*(img))/density # yの重心の位置

        print("N-axis", neutral_axis_x, neutral_axis_y)

        """
        断面二次モーメント (縦)
        I_x = ∫_A y^2 dA
        """
        matrix_x = np.tile(np.abs(self.arange_y - neutral_axis_y), (self.img_shape[0], 1)).T
        I_x = np.sum(matrix_x * img)/self.norm_I_x

        """
        断面二次モーメント (横)
        I_y = ∫_A x^2 dA
        """
        matrix_y = np.tile(np.abs(self.arange_x - neutral_axis_x), (self.img_shape[1], 1))
        I_y = np.sum(matrix_y * img)/self.norm_I_y

        plt.imshow(matrix_x*img/matrix_x.max(), cmap = "gray", vmin=0.0, vmax=1.0)
        plt.show()
        plt.imshow(matrix_y*img/matrix_y.max(), cmap = "gray", vmin=0.0, vmax=1.0)
        plt.show()
        """
        断面二次極モーメント (正規化もする)
        """
        I_r = (I_x + I_y)/2.0

        return I_x, I_y, I_r

    def check(self):

        print(self.matrix_I_x)
        plt.imshow(self.matrix_I_x/self.matrix_I_x.max(), cmap = "gray", vmin=0.0, vmax=1.0)
        plt.show()
        plt.imshow(self.matrix_I_y/self.matrix_I_y.max(), cmap = "gray", vmin=0.0, vmax=1.0)
        plt.show()
        plt.imshow(self.matrix_I_r/self.matrix_I_r.max(), cmap = "gray", vmin=0.0, vmax=1.0)
        plt.show()

def calcMoIoA():
    """
    線画薄い場所＝面積が減っている とする。
    実際のプリントとは異なってしまうが、今回は仕方ない。
    """
    (train_X, train_y), (_, _) = tf.keras.datasets.mnist.load_data()
    train_X = train_X.astype(np.float32).reshape((train_X.shape[0], 28, 28))/255.0

    smoa = SecondMomentOfArea()

    I_x_holder = [[] for i in range(10)]
    I_y_holder = [[] for i in range(10)]
    I_r_holder = [[] for i in range(10)]

    print(f"| Number | I_x mean(var) | I_y mean(var) | I_r mean(var) |")
    for number in range(1, 10):

        number_arr = train_X[train_y == number] # 0, 1, 2, ..., 9

        for i in range(number_arr.shape[0]):
            img = number_arr[i]

            I_x, I_y, I_r = smoa.calc_smoa(img)

            I_x_holder[number].append(I_x)
            I_y_holder[number].append(I_y)
            I_r_holder[number].append(I_r)

            plt.imshow(img, cmap = "gray", vmin=0.0, vmax=1.0)
            plt.show()

        print(f"| {number} | {np.mean(I_x_holder[number]):.3f}({np.std(I_x_holder[number]):.3f}) | {np.mean(I_y_holder[number]):.3f}({np.std(I_y_holder[number]):.3f}) | {np.mean(I_r_holder[number]):.3f}({np.std(I_r_holder[number]):.3f}) |")

    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gray')

    labels = [str(i) for i in range(10)]

    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    ax.boxplot([I_x_holder[i] for i in range(10)], labels=labels, whis="range")
    ax.set_xlabel('Number')
    ax.set_ylabel('I_x')
    ax.set_ylim(0, 1.0)

    ax = fig.add_subplot(3, 1, 2)
    ax.boxplot([I_y_holder[i] for i in range(10)], labels=labels, whis="range")
    ax.set_xlabel('Number')
    ax.set_ylabel('I_y')
    ax.set_ylim(0, 1.0)

    ax = fig.add_subplot(3, 1, 3)
    ax.boxplot([I_r_holder[i] for i in range(10)], labels=labels, whis="range")
    ax.set_xlabel('Number')
    ax.set_ylabel('I_r')
    ax.set_ylim(0, 1.0)
    plt.show()

calcMoIoA()
