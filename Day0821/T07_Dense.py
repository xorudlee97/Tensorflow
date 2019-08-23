import tensorflow as tf

class My_Dense:
    input_Node = None
    W = None
    b = None
    def __init__(self, input_Node = None, output_Node = None, layer_hypothesis, weight_name="weihgt", initializer = None):
        if self.input_Node != None:
            input_Node = self.input_Node;
        else:
            self.input_Node = input_Node;
        if initializer == None:
            W = tf.Variable(tf.random_normal([input_Node, output_Node]), weight_name)
            b = tf.Variable(tf.random_normal([output_Node]), name="bias")
        else:
            W = tf.get_variable(weight_name, shape=[input_Node,output_Node], initializer=initializer)
            b = tf.Variable(tf.random_normal([output_Node]), name="bias")