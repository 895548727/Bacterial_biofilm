import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from datautil import build_all_dataset,generate_num,batch_yield,gettestdata,getdevdata,process_all_data,getfinaldata
import matplotlib.pyplot as plt
import numpy as np
# from flask import Flask, request, Response, jsonify
# from Bio import SeqIO

MAX_LEN=1000
MAX_EPISODES=2001
BATCH_SIZE=50
INPUT_SIZE=20
# process_all_data()
value_dict={"G":0,"A":1,"V":2,"L":3,"I":4,"P":5,"F":6,"Y":7,"W":8,"S":9,"T":10,"C":11,
           "M":12,"N":13,"Q":14,"D":15,"E":16,"K":17,"R":18,"H":19}

def Seq2Mat(seq,max_len):
    nums=[]
    usable=True
    for alpha in seq[:-1]:
        num = [0] * INPUT_SIZE
        if alpha in value_dict.keys():
            num[value_dict[alpha]] = 1
        # else:
        #     usable=False
        #     break
        nums.append(num)
    if usable:
        if len(nums)<max_len:
            nums.extend([[0]*INPUT_SIZE]*(max_len-len(nums)))#pad with zero matrix
        elif len(nums)>max_len:
            nums=nums[:max_len]
    return nums,usable

class CNNnet:
    def __init__(self,input_len,begin,input_size):
        self.input_len=input_len
        self.sess=tf.Session()
        self.input_size=input_size
        self.buildnet()
        if begin:
            self.init()
        else:
            self.restore()

    def buildnet(self):
        self.tf_x=tf.placeholder(tf.float32,[None,self.input_len,self.input_size])
        conv1=tf.layers.conv1d(inputs=self.tf_x,filters=16,kernel_size=4,strides=2,padding='same',activation=tf.nn.relu)
        pool1=tf.layers.max_pooling1d(conv1,pool_size=2,strides=2)
        conv2=tf.layers.conv1d(pool1,32,8,1,'same',activation=tf.nn.relu)
        pool2=tf.layers.max_pooling1d(conv2,5,5)#
        conv3 = tf.layers.conv1d(pool2, 64, 12, 1, 'same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(conv3, 3, 3)  #
        shape=pool3.shape
        flat=tf.reshape(pool3,[-1,int(shape[-1]*shape[-2])]) #?
        FClayer=tf.layers.dense(flat,20)
        self.output=tf.layers.dense(FClayer,6)
        self.tf_y = tf.placeholder(tf.int32, [None, ])
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tf_y, logits=self.output))  # ?
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def predict(self,inputseq):
        out=self.sess.run(tf.nn.softmax(self.output, axis=1),feed_dict={self.tf_x:inputseq})
        print(self.loss)
        return out

    def init(self):
        init_op = tf.group(tf.global_variables_initializer())
        self.sess.run(init_op)

    def trainnet(self):
        print("trainbegin")
        inputs,labels=build_all_dataset()
        self.losses=[]
        self.accuracies=[]
        for step in range(MAX_EPISODES):
            nums=generate_num(BATCH_SIZE,len(inputs)-1)
            batch_inputseqs,batch_labels=batch_yield(nums,inputs,labels)
            batch_inputs=[]
            batch_labels_=[]
            Seq2Mat(batch_inputseqs[49], MAX_LEN)
            for i in range(len(batch_inputseqs)):
                seq_input,seq_usable=Seq2Mat(batch_inputseqs[i],MAX_LEN)
                if seq_usable:
                    batch_inputs.append(seq_input)
                    batch_labels_.append(batch_labels[i])
            batch_inputs=np.array(batch_inputs)
            batch_labels_=np.array(batch_labels_)
            loss,_=self.sess.run([self.loss,self.optimizer], feed_dict={self.tf_x: batch_inputs, self.tf_y: batch_labels_})
            if step%50==0:
                test_inputseqs,test_labels=gettestdata()
                test_inputs=[]
                test_labels_=[]
                for i in range(len(test_inputseqs)):
                    seq_input,seq_usable=Seq2Mat(test_inputseqs[i],MAX_LEN)
                    if seq_usable:
                        test_inputs.append(seq_input)
                        test_labels_.append(test_labels[i])
                test_outputs=self.sess.run(tf.argmax(self.output,axis=1),feed_dict={self.tf_x:test_inputs})
                error_num=0
                error_num1=0
                for i in range(int(len(test_outputs)*0.5)):
                    if(test_outputs[i]!=test_labels_[i]):
                        error_num1+=1
                # accuracy_biofilm = (int(len(test_outputs)*0.5) - error_num1) / int(len(test_outputs)*0.5)
                for i in range(len(test_outputs)):
                    if(test_outputs[i]!=test_labels_[i]):
                        error_num+=1
                accuracy=(len(test_outputs)-error_num)/len(test_outputs)
                TP = int(len(test_outputs) * 0.5) - error_num1
                TN = int(len(test_outputs) * 0.5) + error_num1 - error_num
                FP = error_num1
                FN = error_num - error_num1
                print("TP={}, FP={}, TN={}, FN={}".format(TP,FP,TN,FN))
                # print("SE:TP/(TP+FN)=",TP/(TP+FN+1))
                self.accuracies.append(accuracy)
                self.losses.append(loss)
                # print("SP:TN/(TN+FP) ", TN/(TN+FP))
                print("test: ACC(accuracy)=",accuracy)
                # print("test: MCC=",(TP*TN-FP*FN)/((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP)+1)**(1/2))
                print("test: loss=", loss)
                self.save()

    def plot(self):
        x1=np.arange(1,len(self.losses)+1)
        y1=self.losses
        x2 = np.arange(1, len(self.accuracies) + 1)
        y2 = self.accuracies
        for i in range(len(x1)):
            x1[i]=x1[i]*50
            x2[i] = x2[i] * 50
        plt.xlabel("times")
        plt.ylabel("loss")
        plt.plot(x1, y1, marker='o')
        plt.show()
        plt.xlabel("times")
        plt.ylabel("accuracy")
        plt.plot(x2,y2,marker='o')
        plt.show()

    def save(self):
        saver=tf.train.Saver()
        saver.save(self.sess,r'./model/model.ckpt')

    def restore(self):
        saver=tf.train.Saver()
        saver.restore(self.sess,r'./model/model.ckpt')

    def dev(self):
        dev_inputseqs, dev_labels = getdevdata()
        dev_inputs = []
        dev_labels_=[]
        for i in range(len(dev_inputseqs)):
            seqinput,seq_usable=Seq2Mat(dev_inputseqs[i],MAX_LEN)
            if seq_usable:
                dev_inputs.append(seqinput)
                dev_labels_.append(dev_labels[i])
        dev_outputs = self.sess.run(tf.argmax(self.output, axis=1), feed_dict={self.tf_x: dev_inputs})
        error_num = 0
        for i in range(len(dev_inputs)):
            if (dev_outputs[i] != dev_labels_[i]):
                error_num += 1
        accuracy = (len(dev_inputs) - error_num) / len(dev_inputs)
        print("dev: accuracy=", accuracy)

    def test(self):
        dev_inputseqs = getfinaldata()
        print(len(dev_inputseqs))
        dev_inputs = []
        for i in range(len(dev_inputseqs)):
            seqinput, seq_usable =Seq2Mat(dev_inputseqs[i],MAX_LEN)
            if seq_usable:
                dev_inputs.append(seqinput)
        print(len(dev_inputs))
        dev_outputs = self.sess.run(tf.argmax(self.output, axis=1), feed_dict={self.tf_x: dev_inputs})
        print(dev_outputs)
        return dev_outputs

def main():
    net=CNNnet(MAX_LEN,True,INPUT_SIZE)
    ### for online
    # net.restore()
    # a = net.test()
    # return a
#     # # predict_biofilm(data,model)
    # a = read_original_data()
    # print(a)
    # b,c = Seq2Mat(a,MAX_LEN)
    # # print(b)
    # dev_inputs.append(b)
    # net.predict(dev_inputs)
    net.trainnet()
    # net.dev()
    net.plot()

if __name__=="__main__":
    main()
    # a = main()
    # print(len(a))
    # b = []
    # for i in a:
    #     b.append(str(i))
    # c = [i for i in b if i == "1"]
    # print(len(c))
    # f = open('./data/user/result.txt','w')
    # for line in b:
    #     f.write(line+'\n')
    # f.close()

### 写入文件
# sep = ','
# file = open('list.txt','w')
# file.write(sep.join(b))
# file.close()
### declare a flask app

# app = Flask(__name__)
# @app.route('/')
# def home():
#     return '<html>welcome to home!</html>'
# @app.route('/predict')
# def predict():
#     data = request.json
#     print(data)
#     return "hello zhangzhiyuan"
# app.run(host='127.0.0.1',port=8067)

# model_save_path = './model'
# model_name = 'model'
# @app.route("/predict",methods=['POST'])
# def zhangzhiyuan():
#     receiced_data = request.files['input_seq']
#     # receiced_dirpath = '../views/images'
#     # saver = tf.train.Saver()
#     # session = tf.Session()
#     # ckpt = tf.train.get_checkpoint_state(model_save_path)
#     with tf.Session() as sess:
#         saver = tf.train.import_meta_graph('./model/model.ckpt.meta') ### 加载图模型
#         saver.restore(sess, tf.train.latest_checkpoint('./model'))
#
#
#
#
#     if ckpt and ckpt.model_checkpoint_path:
#         model = saver.restore(session,ckpt.model_checkpoint_path)
#     print("load model successfully" + ckpt.model_checkpoint_path)
# """
# 此文件可以把ckpt模型转为pb模型
# """
#
# import tensorflow as tf
# # from create_tf_record import *
# from tensorflow.python.framework import graph_util
#
#
# def freeze_graph(input_checkpoint, output_graph):
#     '''
#     :param input_checkpoint:
#     :param output_graph: PB模型保存路径
#     :return:
#     '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
#     # 直接用最后输出的节点，可以在tensorboard中查找到，tensorboard只能在linux中使用
#     output_node_names = "dense_1/kernel/Adam"
#     saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
#     graph = tf.get_default_graph()  # 获得默认的图
#     input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
#
#     with tf.Session() as sess:
#         saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
#         output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
#             sess=sess,
#             input_graph_def=input_graph_def,  # 等于:sess.graph_def
#             output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开
#
#         with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
#             f.write(output_graph_def.SerializeToString())  # 序列化输出
#         print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
# #
# #
# # # input_checkpoint='inceptionv1/model.ckpt-0'
# # # out_pb_path='inceptionv1/frozen_model.pb'
# #
# input_checkpoint = 'model/model.ckpt'
# out_pb_path = 'model/frozen_model.pb'
# freeze_graph(input_checkpoint, out_pb_path)
#
# import tensorflow.compat.v1 as tf
# from tensorflow.python.training.py_checkpoint_reader import NewCheckpointReader
# import os
# checkpoint_path = os.path.join(r'./model/model.ckpt')
# reader = NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print('tensor_name:',key)
