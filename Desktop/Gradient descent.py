import numpy as np
import matplotlib.pyplot as plt

# Đạo hàm (Gradient) của hàm mục tiêu: f'(x) = 6x + 2 + 4*cos(x)
def grad(x):
    return 6*x + 2 + 4*np.cos(x)

# Hàm chi phí (Cost function): f(x) = 3x^2 + 2x + 4*sin(x)
def cost(x):
    return 3*x**2 + 2*x + 4*np.sin(x)

# Hàm Gradient Descent với tốc độ học (eta) và giá trị khởi tạo (x0)
def myGD1(eta, x0):
    x = [x0]  # Danh sách để lưu lại các giá trị của x trong các bước lặp
    for it in range(100):
        # Cập nhật x mới theo công thức Gradient Descent
        x_new = x[-1] - eta * grad(x[-1])
        
        # Kiểm tra điều kiện dừng: nếu gradient nhỏ hơn ngưỡng (1e-3)
        if abs(grad(x_new)) < 1e-3:
            break
        
        # Thêm giá trị x mới vào danh sách
        x.append(x_new)
    
    return (x, it)

# Thực thi với hai giá trị khởi tạo khác nhau
(x1, it1) = myGD1(0.1, -5)  # x0 = -5
(x2, it2) = myGD1(0.1, 5)   # x0 = 5

# In kết quả
print('x1 = %f, cost = %f, after %d iterations' % (x1[-1], cost(x1[-1]), it1))
print('x2 = %f, cost = %f, after %d iterations' % (x2[-1], cost(x2[-1]), it2))

# Vẽ đồ thị hàm chi phí để quan sát
x_vals = np.linspace(-10, 10, 100)
y_vals = cost(x_vals)

plt.plot(x_vals, y_vals)
plt.scatter(x1[-1], cost(x1[-1]), color='red')  # Vị trí x1 tối ưu
plt.scatter(x2[-1], cost(x2[-1]), color='blue') # Vị trí x2 tối ưu
plt.title('Cost function')
plt.xlabel('x')
plt.ylabel('cost(x)')
plt.show()
