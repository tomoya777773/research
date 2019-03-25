import numpy as np
import math
import matplotlib.pyplot as plt

length=0.2
m=1
siguma=0.3
a=0.02

class Find_position:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        # self.x = np.array([X[X.shape[0]-1]]).T
        self.num = X.shape[0]

    def direction_func(self):
        """カーネル関数とカーネルの微分関数を定義"""
        kernel_func = lambda a,b : pow(np.linalg.norm(a-b)**2 + length**2 , -1/2)
        diff_kernel_func = lambda a,b,c,d : -(c-d) * pow(np.linalg.norm(a-b)**2 + length**2 , -3/2)

        """ヘッセ行列の項別の関数を定義"""
        diff_xx = lambda a,b,c,d: 3*(c-d)**2 * pow(np.linalg.norm(a-b)**2 + length**2, -5/2) - pow(np.linalg.norm(a-b)**2 + length**2, -3/2)
        diff_xy = lambda a,b,c,d: 3*(c[0]-d[0])*(c[1]-d[1])*pow(np.linalg.norm(a-b)**2 + length**2, -5/2)

        """カーネルベクトル作成"""
        kernel_x = [kernel_func(self.X[i], self.X[self.num -1]) for i in range(self.num)]
        kernel_x = np.array(kernel_x).T
        # print("kernel:", kernel_x)

        """カーネルの微分ベクトル作成"""
        diff_kernel_x = np.zeros((self.num,self.X.shape[1]))
        for i in range(self.num):
            for j in range(self.X.shape[1]):
                diff_kernel_x[i][j] = diff_kernel_func(self.X[self.num-1], self.X[i], self.X[self.num-1][j], self.X[i][j])
        
        # print("diff_kernel:", diff_kernel_x)

        """カーネル行列作成"""
        Kernel_x = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(self.num):
                Kernel_x[i][j] = kernel_func(self.X[i], self.X[j])  #K
        # print(Kernel_x)

        """G, bを求める"""
        G = Kernel_x + siguma**2 * np.identity(self.num)
        invG = np.linalg.inv(G)
        b = np.dot(invG, self.Y - m)
        # print("b:", b)

        """ヘッセ行列作成し，固有値を求める"""
        hesse00 = np.array([diff_xx(self.X[i], self.X[self.num-1], self.X[self.num-1][0], self.X[i][0]) for i in range(self.num)])
        hesse01 = np.array([diff_xy(self.X[i], self.X[self.num-1], self.X[self.num-1], self.X[i]) for i in range(self.num)])
        hesse10 = hesse01.copy()
        hesse11 = np.array([diff_xx(self.X[i], self.X[self.num-1], self.X[self.num-1][1], self.X[i][1]) for i in range(self.num)])
        # print(hesse00)

        Hesse00 = np.dot(hesse00, b)[0]
        Hesse01 = np.dot(hesse01, b)[0]
        Hesse10 = np.dot(hesse10, b)[0]
        Hesse11 = np.dot(hesse11, b)[0]

        Hesse_matrix = np.array([[Hesse00,Hesse01], [Hesse10,Hesse11]])
        # print(Hesse_matrix)
        la, v = np.linalg.eig(Hesse_matrix)
        # print(la)


        """平均と分散"""
        mean = m + np.dot(kernel_x.T, b)
        var = 1/length - np.dot(np.dot(kernel_x.T, invG), kernel_x)

        """平均と分散の微分"""
        self.diff_mean = (np.dot(b.T, diff_kernel_x)).T
        self.diff_var = (- 2 * np.dot(np.dot(kernel_x.T, invG), diff_kernel_x)).T
        # print("diff_mean", self.diff_mean)

        """法線"""
        normal = self.diff_mean / np.linalg.norm(self.diff_mean)

        """接平面"""
        projection_n = np.identity(self.X.shape[1]) - np.dot(normal, normal.T)
        # print(projection_n)

        """新しい方向"""
        S = np.dot(projection_n, self.diff_var)
        # new_direction = a * S / np.linalg.norm(S)
        # print(new_direction)
        # print(np.dot(projection_n, new_direction))

        new_direction = S / np.linalg.norm(S)
        # print(new_direction)
        # print(a * np.dot(projection_n, new_direction))
       

        """新しい位置"""
        # position = self.x + np.dot(projection_n, new_direction)

        # pred_position = np.zeros(2)
        # pred_position[0] = position[0][0]
        # pred_position[1] = position[1][0]

        return la, normal, np.dot(projection_n, new_direction)

if __name__=='__main__':
    print('##################')
    X = np.array([[0.5,1.1], [0.5, 0.479425538604203], [0.49, 0.470625888171158]])
    Y = np.array([[-1],[0]])

    """接触するまで直進"""
    cnt = 1
    while cnt < 1000:
        match_sin = X[X.shape[0]-1][1] - np.sin(X[X.shape[0]-1][0])
        # print(match_sin)
        if match_sin <= 0:
            # Y = np.delete(Y, -1, 0)
            Y = np.append(Y, [0])
            Y = Y[:, np.newaxis]
            # X = np.delete(X, -1, 0)
            print(X)
            break

        else:
            Y = np.append(Y, [-1])
            Y = Y[:, np.newaxis]
            X = np.append(X, [[X[X.shape[0]-1][0], X[X.shape[0]-1][1]-0.002]], axis=0)
            print(X)
        # print([X[X.shape[0]-1][0], X[X.shape[0]-1][1]])
        cnt += 1
        print("cnt", cnt)

    # print(match_sin)
    # print(X.shape)
    # print(Y.T)
    print("------接触-----")

    """探索開始"""
    count = 0
    for i in range(50):
        count += 1
        print("count", count)

        match_sin = X[X.shape[0]-1][1] - np.sin(X[X.shape[0]-1][0])
        print("match:",match_sin)

        fp = Find_position(X, Y)
        pred = fp.direction_func()
        eigenvalue = pred[0]
        normal = pred[1]
        direction = pred[2]
        print("eigenvalue:", eigenvalue)
        print("normal:", normal.T[0])
        print("direction:", direction)

        """凸形状の場合"""
        if eigenvalue[0]>0 and eigenvalue[1]>0:

            new_position = np.array([X[X.shape[0]-1]])[0] + a * direction
            print(np.array([X[X.shape[0]-1]])[0])
            print(new_position)
            X = np.append(X, [new_position], axis=0)

            match_sin = X[X.shape[0]-1][1] - np.sin(X[X.shape[0]-1][0])

            if round(match_sin, 2) == 0:
                Y = np.append(Y, [0])
                Y = Y[:, np.newaxis]
                print("match:", match_sin)

            else:
                Y = np.append(Y, [-1])
                Y = Y[:, np.newaxis]

                cnt = 1
                while cnt < 100:
                    # print("X:", X[X.shape[0]-1])
                    # print("normal:", normal.T[0])
                    normal_position = X[X.shape[0]-1] + 0.002 * cnt * normal.T[0]
                    match_sin = normal_position[1] - np.sin(normal_position[0])
                    # print(match_sin)
                    # print("cnt:", cnt)
                    
                    if round(match_sin, 2) == 0:
                        # X = np.append(X, [X[X.shape[0]-1] + 0.001 * (cnt-1) * normal.T[0]], axis=0)
                        X = np.append(X, [normal_position], axis=0)
                        Y = np.append(Y, [0])
                        Y = Y[:, np.newaxis]
                        break
                    cnt+=1
            # print(Y.T)
            # print(X)
            print("###############################################")


            """凹形状の場合"""
        else:
            normal_position = X[X.shape[0]-1] - 0.02 * normal.T[0]
            X = np.append(X, [normal_position], axis=0)
            Y = np.append(Y, [-1])
            Y = Y[:, np.newaxis]

            cnt = 1
            while cnt < 50:
                new_position = X[X.shape[0]-1] + cnt * 0.002 * direction
                match_sin = new_position[1] - np.sin(new_position[0])
                if (match_sin, 2) == 0:
                    X = np.append(X, [new_position], axis=0)
                    Y = np.append(Y, [0])
                    Y = Y[:, np.newaxis]
                    break
                # print("cnt:", cnt)
                cnt += 1

            match_sin = X[X.shape[0]-1][1] - np.sin(X[X.shape[0]-1][0])

            if (match_sin, 2) != 0:
                X = np.append(X, [new_position], axis=0)
                Y = np.append(Y, [-1])
                Y = Y[:, np.newaxis]
                cnt = 1
                while cnt < 100:
                    normal_position = X[X.shape[0]-1] + 0.002 * cnt * normal.T[0]
                    match_sin = normal_position[1] - np.sin(normal_position[0])
                    if round(match_sin, 2) == 0:
                        X = np.append(X, [normal_position], axis=0)
                        # X = np.append(X, [X[X.shape[0]-1] + 0.002 * (cnt-1) * np.array([0, 1])], axis=0)
                        Y = np.append(Y, [0])
                        Y = Y[:, np.newaxis]
                        break
                    # print("cnt:", cnt)
                    cnt += 1
    print(Y.T)
    print(X)
    # np.save("X_2d_hesse", X)
    # np.save("Y_2d_hesse", Y)
    plt.plot(X.T[0], X.T[1])
    plt.xlim(-5, 1)
    plt.ylim(-1, 1)
    plt.show()