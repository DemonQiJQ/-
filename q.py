import librosa
import numpy as np
import tensorflow as tf
import os

class CNNConfig(object):
    def __init__(self):
        self.filter_size = [2,3,4,5]
        #self.filter_size = [[2,100],[3,100],[4,100],[5,100]]
        self.num_filters = 64
        self.hidden_dim = 256
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.dropout_keep_prob = 0.7
        self.print_per_batch = 50
        self.save_tb_per_batch = 100

        self.batch_size = 128



def load_files():
    root = "recordings"
    train_dic = {}
    validation_dic = {}
    test_dic = {}
    count = np.zeros(10)
    for r,dirs,files in os.walk(root):
        file_num = len(files) / 10
        train_num = file_num * 0.7
        validation_num = file_num * 0.2
        for filename in files:
            filename_list = filename.split(".")[0].split("_",2)
            file_label = int(filename_list[0])
            if count[file_label] < train_num:
                train_dic.update({os.path.join(root,filename):file_label})
            elif count[file_label] < train_num + validation_num:
                validation_dic.update({os.path.join(root,filename):file_label})
            else:
                test_dic.update({os.path.join(root,filename):file_label})
            count[file_label] = count[file_label] + 1
    return train_dic,validation_dic,test_dic
            

def dense_to_one_hot(input,num):
    x = np.zeros(num)
    x[input] = 1
    return x


def batch_iter(features,labels,batch_size):
    num = len(features)
    if num % batch_size == 0:
        num_batches = num // batch_size
    else:
        num_batches = num // batch_size + 1
    output_features = []
    output_labels = []
    for i in range(num_batches-1):
        batch_features = []
        batch_labels = []
        for j in range(batch_size):
            batch_features.append(features[i * batch_size + j])
            batch_labels.append(labels[i * batch_size + j])
        output_features.append(batch_features)
        output_labels.append(batch_labels)
    batch_features = []
    batch_labels = []
    for j in range(num - (num_batches-1) * batch_size):
        batch_features.append(features[(num_batches-1) * batch_size + j])
        batch_labels.append(labels[(num_batches-1) * batch_size + j])
    output_features.append(batch_features)
    output_labels.append(batch_labels)
    return output_features,output_labels,num_batches

def feed_data(cnn, x_batch, y_batch, dropout_keep_prob):
    return {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.keep_prob: dropout_keep_prob}


#对音频文件特征MFCC进行提取
def read_files(files):
    labels = []
    features = []
    for files,ans in files.items():
        wave,sr = librosa.load(files,mono=True)      #wave:audio time series         sr:sampling rate of y
        label = dense_to_one_hot(ans,10)
        labels.append(label)
        mfcc = librosa.feature.mfcc(wave,sr)
        mfcc = np.pad(mfcc,((0,0),(0,100-len(mfcc[0]))),mode='constant',constant_values=0)
        features.append(np.array(mfcc))
    return np.array(features),np.array(labels)

#对特征向量进行归一化处理
def mean_normalize(features):
    std_value = features.std()
    mean_value = features.mean()
    return (features - mean_value) / std_value

class ASRCNN(object):
    def __init__(self,config,width,height,num_classes):         #20,100
        self.config = config
        self.input_x = tf.placeholder(tf.float32,[None,width,height],name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')

        input_x = tf.transpose(self.input_x,[0,2,1])
        pooled_outputs = []
        for i,filter_size in enumerate(self.config.filter_size):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)
                conv = tf.layers.conv1d(input_x,self.config.num_filters,filter_size,activation=tf.nn.relu)
                pooled = tf.reduce_max(conv,reduction_indices=[1])
                pooled_outputs.append(pooled)
        num_filters_total = self.config.num_filters * len(self.config.filter_size)  #64*4
        pooled_reshape = tf.reshape(tf.concat(pooled_outputs,1),[-1,num_filters_total])

        fc = tf.layers.dense(pooled_reshape, self.config.hidden_dim, activation=tf.nn.relu, name='fc1')
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)
        #分类器
        self.logits = tf.layers.dense(fc, num_classes, name='fc2')
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")
        #损失函数
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)
        #优化器
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        #准确率
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def train(argv=None):
    '''batch = mfcc_batch_generator()
    X, Y = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y  # overfit for now'''
    train_files, valid_files, test_files = load_files()
    train_features, train_labels = read_files(train_files)
    train_features = mean_normalize(train_features)
    print('read train files down')
    valid_features, valid_labels = read_files(valid_files)
    valid_features = mean_normalize(valid_features)
    print('read valid files down')
    test_features, test_labels = read_files(test_files)
    test_features = mean_normalize(test_features)
    print('read test files down')

    width = 20  # mfcc features
    height = 100  # (max) length of utterance
    classes = 10  # digits
    config = CNNConfig()
    cnn = ASRCNN(config, width, height, classes)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join('cnn_model', 'model.ckpt')
    tensorboard_train_dir = 'tensorboard/train'
    tensorboard_valid_dir = 'tensorboard/valid'

    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)
    tf.summary.scalar("loss", cnn.loss)
    tf.summary.scalar("accuracy", cnn.acc)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)

    total_batch = 0
    for epoch in range(config.num_epochs):
        #print('Epoch:', epoch + 1)
        #batch_train = batch_iter(train_features, train_labels)
        x_batches,y_batches,num_batches = batch_iter(train_features, train_labels,config.batch_size)
        #for x_batch, y_batch in batch_train:
        for i in range(num_batches):
            x_batch = x_batches[i]
            y_batch = y_batches[i] 
            total_batch += 1
            feed_dict = feed_data(cnn, x_batch, y_batch, config.dropout_keep_prob)
            session.run(cnn.optim, feed_dict=feed_dict)
            if total_batch % config.print_per_batch == 0:
                train_loss, train_accuracy = session.run([cnn.loss, cnn.acc], feed_dict=feed_dict)
                valid_loss, valid_accuracy = session.run([cnn.loss, cnn.acc], feed_dict={cnn.input_x: valid_features,
                                                                                         cnn.input_y: valid_labels,
                                                                                         cnn.keep_prob: config.dropout_keep_prob})
                print('Steps:' + str(total_batch))
                print(
                    'train_loss:' + str(train_loss) + ' train accuracy:' + str(train_accuracy) + '\tvalid_loss:' + str(
                        valid_loss) + ' valid accuracy:' + str(valid_accuracy))
            if total_batch % config.save_tb_per_batch == 0:
                train_s = session.run(merged_summary, feed_dict=feed_dict)
                train_writer.add_summary(train_s, total_batch)
                valid_s = session.run(merged_summary, feed_dict={cnn.input_x: valid_features, cnn.input_y: valid_labels,
                                                                 cnn.keep_prob: config.dropout_keep_prob})
                valid_writer.add_summary(valid_s, total_batch)

        saver.save(session, checkpoint_path, global_step=epoch)
    test_loss, test_accuracy = session.run([cnn.loss, cnn.acc],
                                           feed_dict={cnn.input_x: test_features, cnn.input_y: test_labels,
                                                      cnn.keep_prob: config.dropout_keep_prob})
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))

# 测试数据准备,读取文件并提取音频特征
def read_test_wave(path):
    files = os.listdir(path)
    feature = []
    features = []
    label = []
    for wav in files:
        # print(wav)
        if not wav.endswith(".wav"): continue
        ans = int(wav[0])        
        wave, sr = librosa.load(os.path.join(path,wav), mono=True)
        label.append(ans)
        # print("真实lable: %d" % ans)
        mfcc = librosa.feature.mfcc(wave, sr)
        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - len(mfcc[0]))), mode='constant', constant_values=0)
        feature.append(np.array(mfcc))   
    features = mean_normalize(np.array(feature))
    return features,label

# 模型加载
def test(path):
    features, label = read_test_wave(path)
    print('loading ASRCNN model...')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('cnn_model/model.ckpt-99.meta')
        saver.restore(sess, tf.train.latest_checkpoint('cnn_model'))  
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        pred = graph.get_tensor_by_name("pred:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        for i in range(0, len(label)):
            feed_dict = {input_x: features[i].reshape(1,20,100), keep_prob: 1.0}
            test_output = sess.run(pred, feed_dict=feed_dict)
            
            print("="*15)
            print("真实lable: %d" % label[i])
            print("识别结果为:"+str(test_output[0]))
        print("Congratulation!")  


if __name__ == "__main__":
    train()
    #test("test")
