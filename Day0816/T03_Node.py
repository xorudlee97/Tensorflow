import tensorflow as tf

# 3.0 노드 생성
node1 = tf.constant(3.0, tf.float32)
# 4.0 노드 생성
node2 = tf.constant(4.0)
# 두개의 노드를 묶는 (루트 노드)에 node3을 넣는다.
node3 = tf.add(node1, node2)

print("node1:", node1, "node2:", node2)
print("node3:", node3)

# 실행
sess = tf.Session()
print("sess.run[node1, node2]:", sess.run([node1, node2]))
print("sess.run[node3]:", sess.run(node3))
'''
====================================================
스칼라 원소
1
벡터   1차 행렬
(1,2) => inputdim = 1, input_shape = (1,)
행렬   2차 행렬
[[1,2]] => input_shape = (2,3)
텐서   3차 행렬
===================================================

위 소스의 구조
     Nod1          Node2
     3.0           4.0
        \         /
         \       /
           Node3
           Add(7)
=================================================
Session 위 소스의 구조를 사람이 볼수 있게 출력
'''