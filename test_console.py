from collections import deque

d = deque(maxlen=3)
d.append(10)   # [10]
d.append(20)   # [10, 20]
d.append(30)   # [10, 20, 30]
d.append(40)  
d.append([])
d.append(12)

print(d)