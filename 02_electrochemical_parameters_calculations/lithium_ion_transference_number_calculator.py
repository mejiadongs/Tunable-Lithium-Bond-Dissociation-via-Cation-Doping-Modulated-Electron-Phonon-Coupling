print("欢迎使用锂离子迁移数计算程序!")

# 输入参数
I0 = float(input("请输入初始电流I0(单位:安培): "))
Iss = float(input("请输入稳态电流Iss(单位:安培): "))
V = float(input("请输入施加的极化电压V(单位:伏特): "))
R0 = float(input("请输入极化前的阻抗R0(单位:欧姆): "))
Rss = float(input("请输入极化后的阻抗Rss(单位:欧姆): "))

# 计算锂离子迁移数
t_plus = Iss * (V - I0*R0) / (I0 * (V - Iss*Rss))

# 输出结果
print(f"\n基于输入的参数,锂离子迁移数t+ = {t_plus:.4f}")