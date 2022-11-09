import csv

import numpy
import pylab


def KalmanFilter(z, n_iter=20):
    # 这里是假设A=1，H=1的情况
    # intial parameters
    sz = (n_iter,)  # size of array
    # Q = 1e-5 # process variance
    Q = 1e-6  # process variance
    # allocate space for arrays
    xhat = numpy.zeros(sz)  # a posteri estimate of x
    P = numpy.zeros(sz)  # a posteri error estimate
    xhatminus = numpy.zeros(sz)  # a priori estimate of x
    Pminus = numpy.zeros(sz)  # a priori error estimate
    K = numpy.zeros(sz)  # gain or blending factor
    R = 0.1 ** 2  # estimate of measurement variance, change to see effect
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    A = 1
    H = 1

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = A * xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = A * P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k] + K[k] * (z[k] - H * xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1 - K[k] * H) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    return xhat


def create_csv(row):
    with open('kalman.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(row)


if __name__ == '__main__':
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.readline().split("	")
    print(text)
    raw_data = list()
    for x in text:
        raw_data.append(float(x))
        print(float(x))

    xhat = KalmanFilter(raw_data, n_iter=len(raw_data))
    create_csv(xhat)
    pylab.plot(raw_data, 'k-', label='raw measurement')  # 测量值
    pylab.plot(xhat, 'b-', label='Kalman estimate')  # 过滤后的值
    print(xhat)
    pylab.legend()
    pylab.xlabel('Iteration')
    pylab.ylabel('ADC reading')
    pylab.show()
