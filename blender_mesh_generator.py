import time
import bpy
import random
import numpy as np; np.random.seed(20200527)
import tensorflow as tf
from skimage import morphology
print(tf.__version__)

def delete_all_obj():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[obj.name])

def create_cube(scale=[1.0, 1.0, 1.0], location=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0]):
        bpy.ops.mesh.primitive_cube_add(size=1.0, calc_uvs=True
                                        , enter_editmode=False, align='WORLD'
                                        , location=location, rotation=rotation
                                        )
        bpy.context.object.scale = scale

def create_beam_1x1(val, beam_length, location, rotation):
    """
    通常はただの 1x1 の梁を生成。ピクセル値が中途半端なとき、部材を確率的に弱くし、均等さを保つ。
    （このヒューリスティックは3Dプリンタと相性が悪かったかもしれない。）
    """
    if(val == 1.0):
        create_cube(scale=[1.0, 1.0, beam_length], location=location, rotation=rotation)
    else:
        _r = random.randint(0, 3)
        if(_r==0):#(0, 1, 2, 3)
            create_cube(scale=[0.5, 1.0, beam_length]
                , location=location+[-0.25, 0.0, 0.0], rotation=rotation)
            create_cube(scale=[0.5, 0.5, beam_length]
                , location=location+[+0.25, +0.25, 0.0], rotation=rotation)
        elif(_r==1):
            create_cube(scale=[0.5, 1.0, beam_length]
                , location=location+[-0.25, 0.0, 0.0], rotation=rotation)
            create_cube(scale=[0.5, 0.5, beam_length]
                , location=location+[+0.25, -0.25, 0.0], rotation=rotation)
        elif(_r==2):
            create_cube(scale=[0.5, 1.0, beam_length]
                , location=location+[+0.25, 0.0, 0.0], rotation=rotation)
            create_cube(scale=[0.5, 0.5, beam_length]
                , location=location+[-0.25, +0.25, 0.0], rotation=rotation)
        elif(_r==3):
            create_cube(scale=[0.5, 1.0, beam_length]
                , location=location+[+0.25, 0.0, 0.0], rotation=rotation)
            create_cube(scale=[0.5, 0.5, beam_length]
                , location=location+[-0.25, -0.25, 0.0], rotation=rotation)
        else:
            raise ValueError("out of range")

class Generator:
    def __init__(self):
        self.NOISE_DIM = 128
        self.FIXED_NOISE_FOR_PREDICT = np.random.normal(0, 1, (1, self.NOISE_DIM))

        self.model = self.build_generator()
        self.model.load_weights("C:/Users/alche/Documents/Blender/model-I_x+75-24000.hdf5")

    def build_generator(self):
        z = z_in = tf.keras.layers.Input(shape=(self.NOISE_DIM, ), name="noise")
        x = tf.keras.layers.Dense(1024)(z)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.Dense(7*7*64)(z)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.Reshape(target_shape=(7, 7, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5)
            , padding='same', strides=(2, 2), use_bias=False, activation=None)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(1, kernel_size=(5, 5)
            , padding='same', strides=(2, 2), use_bias=False, activation=None)(x)
        img = tf.math.tanh(x)
        y = tf.keras.layers.Lambda(lambda x: x, name="generated_image")(img) #
        img = (img + 1.0)/2.0
        I_x, I_y, I_r = tf.reduce_sum(img), tf.reduce_sum(img), tf.reduce_sum(img)
        return tf.keras.Model(inputs=z_in, outputs=[y, I_x, I_y, I_r])

    def generate_number(self):
        batch, _, _, _ = self.model.predict(self.FIXED_NOISE_FOR_PREDICT)
        number = (batch[0, :, :, 0] + 1.0)/ 2.0
        return number

class CreateDanmen:
    def __init__(self):
        """
        Initialie constants.
        """
        self.NUM_X_BOXES = 28
        self.NUM_Y_BOXES = 28
        self.NUM_BOXES = self.NUM_X_BOXES * self.NUM_Y_BOXES
        self.BEAM_LENGTH = 60
        self.SUPPORT_THICKNESS = 12

        """
        Load MNIST.
        """
        (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        X_train = (X_train / 255.0)#.astype(np.int)
        """
        0.75 < x to 1
        0.25 < x < 0.75 = 0.5 (variable)
        x < 0.25 to 0
        """
        if(True):val = X_train[1].reshape((28, 28))
        val = Generator().generate_number()

        val[0.75 <= val] = 1.0
        val[np.where(np.asarray(0.25 < val) & np.asarray(val < 0.75))] = 0.5
        val[val <= 0.25] = 0.0
        self.VAL = val

        print(val)

    def boxloop(self, val, beam_length, offset):
        # All boxes
        for x in range(self.NUM_X_BOXES):
            """
            Initialize all primitive params.
            """
            box_loc = np.empty(shape=[self.NUM_Y_BOXES, 3], dtype=np.float32) # Variable
            box_rot = np.zeros(shape=[self.NUM_Y_BOXES, 3], dtype=np.float32) # No rotation
            for y in range(self.NUM_Y_BOXES):
                if(val[x, y] == 0): continue # pass 0 value
                box_loc[y] = [offset[0]+x,  offset[1]+y,  beam_length/2.0]

                create_beam_1x1(val=val[x, y]
                    , beam_length=beam_length
                    , location=box_loc[y], rotation=box_rot[y]) # Cube

    def create_body(self):
        self.boxloop(val=self.VAL, beam_length=self.BEAM_LENGTH, offset=(0, 0))

        #val_blur = filters.gaussian(self.VAL, sigma=3, mode='nearest')
        """
        画像を反転してピンを作る
        """
        half_mask = np.concatenate([np.zeros(shape=[14, 28]), np.ones(shape=[14, 28])], axis=0)
        support_half = (1 - np.ceil(self.VAL)) * half_mask
        support_erosion = morphology.binary_erosion(support_half, morphology.diamond(2)).astype(np.float32)

        for i in range(2):
            self.boxloop(val=support_erosion, beam_length=self.SUPPORT_THICKNESS, offset=(14*i, 0))


if __name__ == "__main__":
    delete_all_obj()

    start_time = time.time()
    createdanmen = CreateDanmen()
    createdanmen.create_body()
    print(f"SpentTime: {time.time() - start_time}[s]")
