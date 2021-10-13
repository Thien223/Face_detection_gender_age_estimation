<<<<<<< HEAD
import tensorflow as tf
print("GPUs:", len(tf.config.list_physical_devices('GPU')))
=======
def solution(n,recipes, orders):
    n=3
    recipes=['A 2','B 3']
    orders = ['A 1','A 2','B 3','B 4']
    
    time = 0
    while True:
        time+=1
        if len(orders)<1:
            break
            
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
