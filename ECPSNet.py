import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 读取数据
df_data = pd.read_excel('C:/Users/1/Desktop/Master/VS code/bridge/xielaqiao.xlsx')  # 假设输出变量在第一列
data = np.array(df_data)  # 100个样本，7列数据（5列输入，2列输出）

# 分割输入和目标数据
input_data = data[:, :5]  # 取前5列作为输入
target_data = data[:, 6:]  # 取后2列作为目标

# 1. 归一化处理
scaler_input = MinMaxScaler(feature_range=(-1,1))  # 用于输入数据的归一化
scaler_target = MinMaxScaler(feature_range=(-1,1))  # 用于输出数据的归一化

# 对输入数据进行归一化
input_data_scaled = scaler_input.fit_transform(input_data)  # 输入数据归一化到 [0, 1]

# 对输出数据进行归一化
target_data_scaled = scaler_target.fit_transform(target_data)  # 输出数据归一化到 [0, 1]



input_window = []
output_window = []
#生成滑动窗口
for i in range(0,len(input_data_scaled)-24):
    input = []
    output = []
    for ii in range(24):
        input.append(list(input_data_scaled[i+ii]))
        output.append(list(target_data_scaled[i+ii]))
    input_window.append(input)
    output_window.append(output)

# 将列表转为 NumPy 数组
input_windows = np.array(input_window)
output_windows = np.array(output_window)

# 3. 划分数据集（80% 训练集，20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(input_windows, output_windows, test_size=0.2, random_state=42)


#转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)


# 创建 TensorDataset 和 DataLoader
data_set = TensorDataset(X_train_tensor, Y_train_tensor)
data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=True, drop_last=False)

data_set_test = TensorDataset(X_test_tensor, Y_test_tensor)
data_loader_test = DataLoader(dataset=data_set_test, batch_size=1, shuffle=False, drop_last=False)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.Mfc1 = nn.Linear(hidden_size,hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.Mfc2 = nn.Linear(hidden_size,hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)


        self.Mfc3 = nn.Linear(input_size,hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.Mfc4 = nn.Linear(hidden_size,hidden_size)
        self.ln4 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.attention = nn.MultiheadAttention(embed_dim=5, num_heads=1)

        self.Mfc5 = nn.Linear(hidden_size,hidden_size)
        self.ln5 = nn.LayerNorm(hidden_size)

        self.Mfc6 = nn.Linear(hidden_size,hidden_size)
        self.ln6 = nn.LayerNorm(hidden_size)

        self.Mfc7 = nn.Linear(hidden_size,1)

    def link1(self,output):
        out = self.Mfc1(output[:, -1, :])  # 取最后一个时间步的输出
        out = F.sigmoid(out)
        out = self.ln1(out)
        out = self.Mfc2(out)  # 取最后一个时间步的输出
        out = F.sigmoid(out)
        out = self.ln2(out)
        out = self.fc1(out)  # 取最后一个时间步的输出
        return out
    
    def link2(self,atte_output):
        out = self.Mfc3(atte_output[0])  # 取最后一个时间步的输出
        out = F.sigmoid(out)
        out = self.ln3(out)
        out = self.Mfc4(out)  # 取最后一个时间步的输出
        out = F.sigmoid(out)
        out = self.ln4(out)
        out = self.fc2(out)  # 取最后一个时间步的输出
        return out


    def forward(self, x):
        last_time_step = x[:, -1, :].unsqueeze(1)  # (batch_size, 1, input_size)

        # 调整形状以匹配 MultiheadAttention 的输入要求
        last_time_step = last_time_step.permute(1, 0, 2)  # (1, batch_size, input_size)


        # 应用注意力机制
        attn_output, attn_weights = self.attention(last_time_step, last_time_step, last_time_step) 


        output0, (h_n, c_n) = self.lstm1(x)
        output = F.relu(output0)
        attn_output = F.relu(attn_output)
        out1 = self.link1(output)  # 取最后一个时间步的输出
        out2 = self.link2(attn_output)   #用于残差
        out2, (h_n1, c_n1) = self.lstm2(out2)
        a = F.sigmoid(out1)
        out = a + out2    #分层计算后求和
        out = self.ln5(out)
        out = self.Mfc5(out)

        out = out + output0[:, -1, :]   #与历史数据lstm相加
        out = self.ln5(out)
        out = self.Mfc6(out)
        out = F.sigmoid(out)

        out = out + out2   #与注意力结果相加
        out = self.ln6(out)
        out = self.Mfc7(out)
        return out



def R2_c(pingfangcha,shiji):
    ave = sum(shiji)/len(shiji)
    shiji_chafang_list = []
    for i in shiji:
        ave2 = (i-ave)**2
        shiji_chafang_list.append(ave2)
    R2 = sum(pingfangcha)/sum(shiji_chafang_list)
    return 1-R2

def MAPE_c(chazhi_list,shiji):
    MAPE__list = []
    for i,j in zip(chazhi_list,shiji):
        mape = abs(i/j)
        MAPE__list.append(mape)
    return sum(MAPE__list)/len(MAPE__list)



# 实例化模型
input_size = 5
hidden_size = 64
output_size = 1
num_layers = 1

model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.L1Loss() #绝对误差MAE

# 训练模型
num_epochs = 500
MAE_list = []
RMSE_list = []
R2_list = []
MAPE_list = []

for epoch in range(num_epochs):
    chazhi_list = []
    chazhi_pingfang_list = []
    shiji_list = []
    for cnt, batch in enumerate(data_loader):
        train, test = batch
        # train = train.unsqueeze(1)  # 添加时间步维度，形状变为 (batch_size, seq_len, input_size)
        # test = test.unsqueeze(1)  # 添加时间步维度，形状变为 (batch_size, seq_len, output_size)

        # 前向传播
        x = model(train)

        # 计算损失
        loss = loss_fn(x, test[:, -1, :])  # 取最后一个时间步的目标值

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for cnt1, batch1 in enumerate(data_loader_test):
        input1,test1 = batch1
        x = model(input1)
        test1 = test1.cpu().numpy()
        x = x.cpu().detach().numpy()
        x = scaler_target.inverse_transform(x)
        test1 = scaler_target.inverse_transform(test1[:,-1,:])
        chazhi = float(abs(x[0][0] - test1[0][0])) #计算预测值与实际值的差的绝对值
        chazhi_pingfang = chazhi ** 2
        # 记录损失
        chazhi_list.append(chazhi)
        chazhi_pingfang_list.append(chazhi_pingfang)
        shiji_list.append(float(test1[0][0]))

    
    MAE = sum(chazhi_list)/len(chazhi_list)
    RMSE = math.sqrt(sum(chazhi_pingfang_list)/len(chazhi_pingfang_list))
    R2 = R2_c(chazhi_pingfang_list, shiji_list)
    MAPE = MAPE_c(chazhi_list,shiji_list)
    MAE_list.append(MAE)
    RMSE_list.append(RMSE)
    R2_list.append(R2)
    MAPE_list.append(MAPE)
    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{cnt+1}/{len(data_loader)}], MAE: {MAE}, RMSE: {RMSE}, R2:{R2}, MAPE:{MAPE}')



x = []
for i in range(len(MAPE_list)):
    x.append(i)
    

torch.save(model, 'hengxiang.pt')


plt.plot(x,MAE_list,color='blue',label = 'MAE')
plt.plot(x,RMSE_list,color='red',label = 'RMSE')
plt.legend()
plt.show()

plt.plot(x,R2_list,color='blue',label = 'R2')
plt.plot(x,MAPE_list,color='red',label = 'MAPE')
plt.legend()
plt.show()

