def solution(n,recipes, orders):
    n=3
    recipes=['A 2','B 3']
    orders = ['A 1','A 2','B 3','B 4']
    
    time = 0
    while True:
        time+=1
        if len(orders)<1:
            break
            