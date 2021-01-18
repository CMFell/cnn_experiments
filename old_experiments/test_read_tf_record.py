import tensorflow as tf

EPOCHS = 2
BATCH_SIZE = 16

Tfrecs = "E:/CF_Calcs/BenchmarkSets/GFRC/TFrec/gfrc_yolo_v2.tfrecords"

aug_data = tf.data.TFRecordDataset(Tfrecs)

def decode(serialized_example):
    # reader = tf.TFRecordReader()

    # _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'imheight': tf.FixedLenFeature([], tf.int64),
            'imwidth': tf.FixedLenFeature([], tf.int64),
            'imdepth': tf.FixedLenFeature([], tf.int64),
            'filename': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'bxy': tf.FixedLenFeature([], tf.int64),
            'bxx': tf.FixedLenFeature([], tf.int64),
            'nanc': tf.FixedLenFeature([], tf.int64),
            'otln': tf.FixedLenFeature([], tf.int64),
            'mxgt': tf.FixedLenFeature([], tf.int64),
            'gt1_list': tf.VarLenFeature(tf.float32),
            'gt2_list': tf.VarLenFeature(tf.float32)
        })

    # Get meta data
    imheight = tf.cast(features['imheight'], tf.int32)
    imwidth = tf.cast(features['imwidth'], tf.int32)
    channels = tf.cast(features['imdepth'], tf.int32)
    boxy = tf.cast(features['bxy'], tf.int32)
    boxx = tf.cast(features['bxx'], tf.int32)
    nanc = tf.cast(features['nanc'], tf.int32)
    outlen = tf.cast(features['otln'], tf.int32)
    maxgt = tf.cast(features['mxgt'], tf.int32)
    filen = tf.cast(features['filename'], tf.string)


    # Get data shapes
    image_shape = tf.stack([imheight, imwidth, channels])
    gt1_shape = tf.stack([boxy, boxx, nanc, outlen])
    gt2_shape = tf.stack([1, 1, 1, 4, maxgt])

    # Get data
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, image_shape)
    image = tf.to_float(image)
    image = tf.divide(image, 255.0)
    gt1 = tf.sparse_tensor_to_dense(features['gt1_list'], default_value=0.0)
    gt1 = tf.reshape(gt1, gt1_shape)
    gt2 = tf.sparse_tensor_to_dense(features['gt2_list'], default_value=0.0)
    gt2 = tf.reshape(gt2, gt2_shape)
    labels = {'gt1': gt1, 'gt2': gt2, 'fn':filen}

    return {"image": image}, labels

aug_data = aug_data.shuffle(buffer_size=12000)
aug_data = aug_data.map(decode, num_parallel_calls=4)
aug_data = aug_data.batch(BATCH_SIZE)

iterator = tf.data.Iterator.from_structure(aug_data.output_types, aug_data.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(aug_data)

sess = tf.Session()

for i in range(EPOCHS):
    sess.run(training_init_op)
    while True:
        try:
            elem = sess.run(next_element)
            gt = elem[1]
            print(gt['fn'])
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break
    print("Epoch")
    print(i)

