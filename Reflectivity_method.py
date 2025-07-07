"""
@File    : rm.py
@Author  : Pesion
@Date    : 2024/2/29
@Desc    :
"""
import numpy as np
from scipy.special import jv
from scipy.fft import ifft
from scipy import integrate
import matplotlib.pyplot as plt
from util import *

method = 'RM'
CZERO = 0
YI = 1j
HSR = 0.0
degrad = np.arctan(1.0) / 45.0

# vp = np.array([3.200, 2.200, 3.200, 2.200, 3.200])# km/s
# vs = np.array([1.816, 1.300, 1.816, 1.300, 1.816])
# rho = np.array([2.5, 1.5, 2.5, 1.5, 2.5])
# QAI = []  # 1.0/QA (ATTENUATION FOR P)
# QBI = []  # 1.0/QB (ATTENUATION FOR S)
# thick = np.array([0.20, 0.008, 0.150, 0.008, 0.2])*1000 # km

QAI = []  # 1.0/QA (ATTENUATION FOR P)
QBI = []  # 1.0/QB (ATTENUATION FOR S)
# 深度域
vp = np.array([3200, 2200, 3200, 2200, 3200]) / 1000
vs = np.array([1816, 1300, 1816, 1300, 1816]) / 1000
rho = np.array([2500, 1500, 2500, 1500, 2500]) / 1000
thick = np.array([200, 8, 150, 8, 200]) / 1000
dt = 0.0001
dz = 0.00001
theta_d = np.arange(0, 31, 5, dtype=int)
theta_r = np.radians(theta_d)
time, vp_t, vs_t, rho_t = depth2time(vp, vs, rho, thick, dt)
layer, vp_z, vs_z, rho_z, thick = time2depth(vp_t, vs_t, rho_t, dt, 0.02, dz)
Nt = len(time)

# 半空间自由平面

# 慢度数/射线数
NRP = 600

# control parameter
NNF = 0  # 0-近场和远场项 1-远场：（贝塞尔函数的渐近形式）

# slowness parameter
pmin = 0.0001
pmax = 0.5001
# Slowness taper
ISW8 = 10  # 在慢度窗口的下端逐渐变小
ISW9 = 10  # 在慢度窗口的上端逐渐变小

iexp = True  # 指数阻尼
NRS = 1  # 指数时间阻尼或无时间阻尼？？

# Nt = 10000

df = 1 / (dt * Nt)  # 什么意思？

# 频率滤波器设置

# Frequency
# taper(low)
fl1 = 0.01
fl2 = 0.10

# taper(high)
fu1 = 10.0
fu2 = 20.0

# Dominant frequency主频
f0 = 2

# 源所在层位
JS = 0

# 源的深度 (from surface)
HSS = 3

#  Distances and azimuths
NPX = 10
XS = np.arange(2, 2 * NPX + 1, 2)  # 距离/km
AZ = np.radians(np.full(NPX, 45))  # 10个45度角的

# 计算积分参数
PG = []  # 慢度序列，长度等于慢度点数
DP = []  # 慢度梯度？？
CMX = 999.999
if pmin > 0:
    CMX = 1.0 / pmin
CMN = 1.0 / pmax
# RP = float(NRP)
DPC = (pmax - pmin) / NRP  # DPC为慢度采样步长
pmin = 0.0001  # 如果pmin为0时，给定一个小值
PG.append(pmin)  # PG加入最小慢度为首元素
PC = pmin + DPC

# IF (ISW(10).NE.3) then 如果是T-X  OUTPUT
# ----------------------------------------------TODO按照一定方法填满PG
Q0 = np.arcsin(pmin - 0.2)
Q1 = np.arcsin(pmax - 0.2)
QMN = 0.5 * Q0 + 0.25 * np.sin(2.0 * Q0)
QMX = 0.5 * Q1 + 0.25 * np.sin(2.0 * Q1)
PQR = (pmax - pmin) / (QMX - QMN)
DP.append(DPC * PQR * np.sqrt(1.0 - (pmin - 0.2) ** 2))

for MK in range(1, NRP):
    Q = np.arcsin(PC - 0.2)
    QP = 0.5 * Q + 0.25 * np.sin(2.0 * Q)
    PG.append(pmin + (QP - QMN) * PQR)
    DP.append(PG[MK] - PG[MK - 1])
    PC = PC + DPC

FD = fu2 - fl1
FR = 0.0
# -----------------------------------------
## 构建子波，通过频率限制

# TODO

# Reduced Time window
PR = 0  # Reduction slowness
STMIN = 0  # start time min

# Debug options ISW(14) ISW(15)

# read velocity

NLA = len(thick)
"""
number of layer 层数
"""

ndt = 0
"""
采样时间，这里使用的是深度域，故为0
"""
NL = NLA + 1  # 界面有nla+1个

"""
The index nr controls the number of reverberations in the layer

nr = 0  - no reflection from top of layer 顶层没有反射
   = 1  - no internal multiples in layer 内部无叠加
  >= 3  - all internal multiples in layer 考虑内部混响
"""
NR = [3, 3, 3, 3, 3, 3,3,3,3,3,3]  # 每一层计算混响的方式
RHO = np.insert(rho, 0, rho[0])
HL = np.insert(thick, 0, thick[0])
# RHO = [rho[0]] + rho  # 每层的密度
# HL = [0] + thick  # 每层的厚度，第一层设厚度为0

# 考虑Q时需要计算vp*vp*CQA*CQA
# CQA = 1.0-0.5*YI*QAI
# CQB = 1.0-0.5*YI*QBI

# 目前不考虑Q
ALSQ = [vp[0] ** 2] + [v ** 2 for v in vp]  # vp方
BESQ = [vs[0] ** 2] + [v ** 2 for v in vs]  # vs方
vp = np.insert(vp, 0, vp[0])
vs = np.insert(vs, 0, vs[0])
Z1 = 0

# 获取源所在层的速度平方、密度平方
# ALFSSQ = ALSQ[JS]
# BETSSQ = BESQ[JS]
# RHOS = RHO[JS]
# NSA = NL - JS  # 源下还有的层数
# JSA = JS - 1  # 源上的层序号

# 通过rface赋值
RDPP = []
RDPS = []
RDSP = []
RDSS = []
RDHH = []
RUHH = []
TDHH = []
TUHH = []

TDPP = []
TDPS = []
TDSP = []
TDSS = []
TUPP = []
TUPS = []
TUSP = []
TUSS = []

RUPP = []
RUPS = []
RUSP = []
RUSS = []

HP = []  # thick*q_vp
HS = []  # thick*q_vs

# ED = []  # 相位收入矩阵

XL = [-1.0, 1.0, 1.0, -1.0, -1.0]

RU_ = np.zeros([5, 2500, 600])  # [order, p, frequency]
RV = np.zeros([5, 2500, 600])
RW = np.zeros([5, 2500, 600])
Omega1 = np.arange(1,Nt) * 2 * np.pi / (Nt * dt)

def respose():
    XS = np.arange(1, 10 + 1, 1)  # 距离/km 偏移距
    # TODO 没细看
    Omega = np.arange(0,Nt) * 2 * np.pi / (Nt * dt)
    DW = 2 * np.pi * df  # 2pif 角频率间隔
    NLO = int(fl1 / df + 0.1) + 1
    NLP = NLO - 1
    NUP = int(fu2 / df + 0.1) + 1
    NUQ = NUP + 1
    NTM = Nt / 2 + 1
    NW = NUP - NLO + 1
    if NRS > 0:
        EPS = 0
    else:
        EPS = 4.0 * df
    # AMUF = RHO[0] * BESQ[0]  # 表面刚度
    # AIJS = 1.0 / ALFSSQ  # 源处慢度平方
    # BIJS = 1.0 / BETSSQ  # 源处慢度平方

    R_wp = []
    R_xw1 = np.zeros([NUP - NLO, 10], np.complex64)
    theta = np.arcsin([P * vp[0] for P in PG], dtype=np.complex64)
    for LP, itheta in enumerate(theta_r):  # 慢度循环
        # PQ = P * P
        # QAS = np.sqrt(AIJS - PQ)  # 源处的垂直慢度
        # QBS = np.sqrt(BIJS - PQ)  # 源处的垂直慢度
        # YAS = -2.0 * PQ + BIJS  # -2p^2+源处横波速度倒数平方
        # QAJF, QBJF = rface(P, PQ)  # 层位循环
        # 通过rface赋值
        RDPP.clear()
        RDPS.clear()
        RDSP.clear()
        RDSS.clear()
        RDHH.clear()
        RUHH.clear()
        TDHH.clear()
        TUHH.clear()

        TDPP.clear()
        TDPS.clear()
        TDSP.clear()
        TDSS.clear()
        TUPP.clear()
        TUPS.clear()
        TUSP.clear()
        TUSS.clear()

        RUPP.clear()
        RUPS.clear()
        RUSP.clear()
        RUSS.clear()
        RD, RU, TD, TU = rface(itheta)
        W = np.pi * 2 * fl1 + YI * EPS  # 后面加了个1j*EPS表示阻尼
        LW = 1
        R_w = []
        R_xw = []
        for f in Omega:  # 频率循环
            R_res = rtdn(f, RD, RU, TD, TU)
            # R_res = R_res.reverse()  # 变成从上到下的反射系数
            R_w.append(R_res)  # 当前频率的反射系数
            # R_cal = W*W*P*R_res[0,0]*jv(0,W*P*XS)
            # R_xw.append(R_cal)

            # W = W + DW
            # LW = LW + 1
        # R_xw = np.array(R_xw)
        # R_xw1 += R_xw
        R_w = np.array(R_w)
        R_wp.append(R_w)  # 当前慢度/角度的反射系数
    R_wp = np.array(R_wp).T
    # Rpp1 = np.flip(np.conj(R_wp[1:359,:]), 0)
    # R_wp = np.concatenate([R_wp[:360,:], Rpp1],0)
    R_pt = ifft(R_wp, axis=0)
    ref = np.flip(R_pt, axis=0).real
    plt.figure()
    freqs = np.fft.fftfreq(Nt, d=dt)
    plt.plot((R_wp[:, 6]))
    wavemat = generate_ricker(Nt, 40, dt)
    obs = wavemat @ ref
    ava(obs,theta_d,time,vp_t,5,method=method,ams=0.4)
    # show_ref(ref[:,:],time[:],6, theta_d, method)
    time_range=0
    if time_range != 0:
        show_ref(ref[int(time_range / dt):-int(time_range / dt), :], time[int(time_range / dt):-int(time_range / dt)], 6, theta_d, method)
    else:
        show_ref(ref, time, 6, theta_d, method)
    avo_time = 0.132
    avo_time2 = 0.233
    avo(obs, theta_d, [int(avo_time / dt), int(avo_time2 / dt)],
        [f'phinney1(t={avo_time}s)', f'phinney2(t={avo_time2}s)'], method)
    # save([obs, ref], 'rm.mat', ['d', 'a'])
    # plt.plot(t, (obs[:, 5]))
    # plt.imshow(obs)
    # avo(obs,theta_d,[133,235],['Phinney1','Phinney2'])
    plt.show()
    return 0




def create_RT(b, r, P, qa, qb):
    """
    :param b: vs^2
    :param r: rho
    :param P: 固定慢度
    :param qa: vp水平慢度
    :param qb: vs水平慢度
    :return:
    """
    epsilon_a = 1 / np.sqrt(2 * r * qa)  # (3.32)
    epsilon_b = 1 / np.sqrt(2 * r * qb)  # (3.33)
    # epsilon_a = 1
    # epsilon_b = 1
    m11 = 1j * qa * epsilon_a
    m12 = P * epsilon_b
    m21 = P * epsilon_a
    m22 = 1j * qb * epsilon_b

    n11 = r * (2 / b * (P ** 2) - 1) * epsilon_a
    n12 = 2j * r / b * P * qb * epsilon_b
    n21 = 2j * r / b * P * qa * epsilon_a
    n22 = r * (2 / b * (P ** 2) - 1) * epsilon_b

    mu = np.array([[-m11, m12], [m21, -m22]], dtype=np.complex64)
    md = np.array([[m11, m12], [m21, m22]], dtype=np.complex64)
    nu = np.array([[n11, -n12], [-n21, n22]], dtype=np.complex64)
    nd = np.array([[n11, n12], [n21, n22]], dtype=np.complex64)

    return mu, md, nu, nd


def rface(itheta):
    """
    计算每层的反射系数
    :param P: 慢度
    :param PQ: 慢度平方
    :return:
    """
    RD = []
    RU = []
    TD = []
    TU = []
    # J为最后一层
    J = NL - 1
    ROJ = RHO[J]
    AIJ = 1.0 / ALSQ[J]
    BIJ = 1.0 / BESQ[J]
    P = np.sin(itheta) / vp[J]
    PQ = P * P
    QAJ = np.sqrt(AIJ - PQ, dtype=np.complex64)
    QBJ = np.sqrt(BIJ - PQ, dtype=np.complex64)
    mu2, md2, nu2, nd2 = create_RT(BIJ, ROJ, P, QAJ, QBJ)  # 根据当前层的慢度,慢度计算特征向量矩阵D,用于计算单层入射的反射系数
    for LK in range(NLA):  # 层位循环
        JB = J - 1  # J的上一层
        DEJ = HL[J]  # J的层厚度
        ROJB = RHO[JB]
        AIJB = 1.0 / ALSQ[JB]
        P = np.sin(itheta) / vp[JB]
        PQ = P * P
        QAJB = np.sqrt(AIJB - PQ, dtype=np.complex64)
        BIJB = 1.0 / BESQ[JB]
        QBJB = np.sqrt(BIJB - PQ, dtype=np.complex64)

        # mu11 = 1j * QAJB
        # mu12 = P
        # mu21 = P
        # mu22 = 1j * QBJB
        #
        # nu11 = ROJ * (2 / BIJB * PQ - 1)
        # nu12 = 2j * ROJ / BIJB * P * QBJB
        # nu21 = 2j * ROJ / BIJB * P * QAJB
        # nu22 = ROJ * (2 / BIJB * PQ - 1)
        #
        # mu_1 = np.array([[mu11, mu12], [mu21, mu22]], dtype=np.complex64)
        # nu_1 = np.array([[nu11, nu12], [nu21, nu22]], dtype=np.complex64)
        mu1, md1, nu1, nd1 = create_RT(BIJB, ROJB, P, QAJB, QBJB)  # JB层的特征矩阵
        deter = mu1.T @ nd2 - nu1.T @ md2  # <mu-,md+>
        R_f = mu1.T @ nu2 - nu1.T @ mu2  # <mu-,mu+>
        denominator = np.linalg.pinv(deter)  # <mu-,md+> -1

        ru = -R_f @ denominator  # (5.29)
        rd = -(md1.T @ nd2 - nd1.T @ md2) @ denominator  # (5.31)(5.32)
        rd1 = -(md2.T @ nd1 - nd2.T @ md1.T) @ np.linalg.pinv(md2.T @ nu1 - nd2.T @ mu1)
        td = 1j * denominator  # (5.34)
        tu = td.T  # (5.29)

        # downward reflection P84? P41
        CJ = 2 * (ROJB / BIJB - ROJ / BIJ)  # 2\delta\mu
        CJPQ = CJ * PQ  # 2\delta\mu p^2
        EJ = CJPQ - ROJB
        FJ = CJPQ + ROJ
        HJ = EJ + ROJ
        RRJ = ROJ * ROJB
        FD = PQ * HJ * HJ + QAJ * (QBJ * EJ * EJ + RRJ * QBJB)
        GD = QAJB * (QBJB * (CJ * CJPQ * QAJ * QBJ + FJ * FJ) + RRJ * QBJ)
        RC = 2 * P * (HJ * FJ + CJ * EJ * QAJ * QBJ)
        DJ = GD + FD

        QQC = (GD - FD) / DJ
        RDPP.append(QQC)
        # RDPPR(LK) = QQR
        # RDPPI(LK) = QQI
        QQC = -QBJB * RC / DJ
        RDPS.append(QQC)
        # RDPSI(LK) = QQI
        QQC = -QAJB * RC / DJ
        RDSP.append(QQC)
        # RDSPR(LK) = QQR
        # RDSPI(LK) = QQI
        QQC = (FD - GD + 2 * RRJ * (QBJ * QAJB - QBJB * QAJ)) / DJ
        RDSS.append(QQC)
        # RDSSR(LK) = QQR
        # RDSSI(LK) = QQI

        # trasmit
        RC = 2 * (EJ * QBJ - FJ * QBJB) / DJ
        QQC = -ROJB * QAJB * RC
        TDPP.append(QQC)
        QQC = -ROJ * QAJ * RC
        TUPP.append(QQC)
        RC = 2 * (EJ * QAJ - FJ * QAJB) / DJ
        QQC = -ROJB * QBJB * RC
        TDSS.append(QQC)
        QQC = -ROJ * QBJ * RC
        TUSS.append(QQC)
        RC = 2 * P * (HJ + CJ * QAJ * QBJB) / DJ
        QQC = ROJB * QAJB * RC
        TDSP.append(QQC)
        QQC = ROJ * QBJ * RC
        TUPS.append(QQC)
        RC = 2 * P * (HJ + CJ * QBJ * QAJB) / DJ
        QQC = -ROJB * QBJB * RC
        TDPS.append(QQC)
        QQC = -ROJ * QAJ * RC
        TUSP.append(QQC)

        # upward reflection
        FD = PQ * HJ * HJ + QAJB * (QBJB * FJ * FJ + RRJ * QBJ)
        GD = QAJ * (QBJ * (CJPQ * CJ * QAJB * QBJB + EJ * EJ) + RRJ * QBJB)
        DJ = GD + FD
        RC = 2 * P * (HJ * EJ + CJ * QAJB * QBJB * FJ)
        QQC = (GD - FD) / DJ
        RUPP.append(QQC)
        QQC = -QBJ * RC / DJ
        RUPS.append(QQC)
        QQC = -QAJ * RC / DJ
        RUSP.append(QQC)
        QQC = (FD - GD - 2 * RRJ * (QAJB * QBJ - QAJ * QBJB)) / DJ
        RUSS.append(QQC)

        # SH波 P83 P44剪切模量计算 mu = rho*beta^2
        EJ = ROJ * QBJ / BIJ
        FJ = ROJB * QBJB / BIJB
        RC = EJ + FJ
        QQC = (FJ - EJ) / RC  # (5.12)(1)
        RDHH.append(QQC)
        RUHH.append(-QQC)  # (5.14)(1)
        QQC = 2 * FJ / RC  # (5.12)(1)??
        TDHH.append(QQC)
        QQC = 2 * EJ / RC  # (5.14)(3)
        TUHH.append(QQC)

        # 相位参数
        HP.append(DEJ * QAJ)  # h*qa
        HS.append(DEJ * QBJ)  # h*qb
        # ED.append(np.exp(np.diag([1j * W * DEJ * QAJ, 1j * W * DEJ * QBJ])))  # (3.45) and (6.24)

        # 向上递归
        QAJ = QAJB
        QBJ = QBJB
        BIJ = BIJB
        ROJ = ROJB
        J = J - 1
        mu2, md2, nu2, nd2 = mu1, md1, nu1, nd1

        # 深层到浅层的反射系数
        # RD.append(np.array([[RDPP[LK], RDPS[LK]], [RDSP[LK], RDSS[LK]]]))  # RDSS正负号相反
        # TD.append(np.array([[TDPP[LK], TDPS[LK]], [TDSP[LK], TDSS[LK]]]))
        # RU.append(np.array([[RUPP[LK], RUPS[LK]], [RUSP[LK], RUSS[LK]]]))  # RUSS正负号相反
        # TU.append(np.array([[TUPP[LK], TUPS[LK]], [TUSP[LK], TUSS[LK]]]))
        RD.append(rd)  # RDSS正负号相反
        TD.append(td)
        RU.append(ru)  # RUSS正负号相反
        TU.append(tu)

    # free surface
    QAJF = QAJ
    QBJF = QBJ
    QMJF = FJ

    return RD, RU, TD, TU
    # return QAJF, QBJF


def rtdn(W, RD, RU, TD, TU):
    """
    叠层计算
    :return:
    """
    J = NL - 1
    R_res = []

    TPP = TDPP[0]
    TPS = TDPS[0]
    TSP = TDSP[0]
    TSS = TDSS[0]
    THH = TDHH[0]

    RPP = RDPP[0]
    RPS = RDPS[0]
    RSP = RDSP[0]
    RSS = RDSS[0]
    RHH = RDHH[0]
    RD_upper = RD[0]
    TD_upper = TD[0]

    if NL < 2:  # 源下小于两层，直接赋值，不需要计算混响
        pass
        # RPP = DCMPLX(RPPR, RPPI)
        # RPS = DCMPLX(RPSR, RPSI)
        # RSP = DCMPLX(RSPR, RSPI)
        # RSS = DCMPLX(RSSR, RSSI)
        # RHH = DCMPLX(RHHR, RHHI)
        # TPP = DCMPLX(TPPR, TPPI)
        # TPS = DCMPLX(TPSR, TPSI)
        # TSP = DCMPLX(TSPR, TSPI)
        # TSS = DCMPLX(TSSR, TSSI)
        # THH = DCMPLX(THHR, THHI)
    else:
        for LK in range(NL - 1):  # 源下开始递归

            # 计算相位收入矩阵

            YDIW = 1j * W
            PHPP = np.exp(YDIW * HP[LK])
            PHSS = np.exp(YDIW * HS[LK])
            ED = np.diag([np.exp(1j * W * HP[LK]), np.exp(1j * W * HS[LK])])
            PHPS = PHPP * PHSS
            PHPP = PHPP * PHPP
            PHSS = PHSS * PHSS

            RPP = PHPP * RPP
            RPS = PHPS * RPS
            RSP = PHPS * RSP
            RSS = PHSS * RSS
            RHH = PHSS * RHH
            RD_upper = ED @ RD_upper @ ED
            TD_upper = TD_upper @ ED

            # 考虑层间混响 layer reverberations 二阶近似

            # UPPR = RUPPR(LK) * RPPR - RUPPI(LK) * RPPI + RUPSR(LK) * RSPR - RUPSI(LK) * RSPI
            # UPPI = RUPPR(LK) * RPPI + RUPPI(LK) * RPPR + RUPSR(LK) * RSPI + RUPSI(LK) * RSPR
            # UPSR = RUPPR(LK) * RPSR - RUPPI(LK) * RPSI + RUPSR(LK) * RSSR - RUPSI(LK) * RSSI
            # UPSI = RUPPR(LK) * RPSI + RUPPI(LK) * RPSR + RUPSR(LK) * RSSI + RUPSI(LK) * RSSR
            # USPR = RUSPR(LK) * RPPR - RUSPI(LK) * RPPI + RUSSR(LK) * RSPR - RUSSI(LK) * RSPI
            # USPI = RUSPR(LK) * RPPI + RUSPI(LK) * RPPR + RUSSR(LK) * RSPI + RUSSI(LK) * RSPR
            # USSR = RUSPR(LK) * RPSR - RUSPI(LK) * RPSI + RUSSR(LK) * RSSR - RUSSI(LK) * RSSI
            # USSI = RUSPR(LK) * RPSI + RUSPI(LK) * RPSR + RUSSR(LK) * RSSI + RUSSI(LK) * RSSR
            #
            # UHHR = RUHHR(LK) * RHHR - RUHHI(LK) * RHHI
            # UHHI = RUHHR(LK) * RHHI + RUHHI(LK) * RHHR

            # 计算(6.15)
            # 从下往上递归
            UPP = RUPP[LK] * RPP + RUPS[LK] * RSP
            UPS = RUPP[LK] * RPS + RUPS[LK] * RSS
            USP = RUSP[LK] * RPP + RUSS[LK] * RSP
            USS = RUSP[LK] * RPS + RUSS[LK] * RSS
            # RR_matrix = np.array([[UPP, UPS], [USP, USS]])
            RR_matrix = RU[LK] @ RD_upper

            # (6.14)
            UHH = RUHH[LK] * RHH

            if NR[J] == 3:  # 逆矩阵的近似情况(6.20) a single internal reverberation
                UPP = 1 + UPP
                USS = 1 + USS
                UHH = 1 + UHH
                RR_matrix = RR_matrix + np.identity(2)  # + RR_matrix@RR_matrix
            elif NR[J] == 2:
                UPP = 1 + UPP
                USS = 1 + USS
                UHH = 1 + UHH
                # 上面的不对，仅供调试
                RR_matrix = np.linalg.pinv(np.identity(2) - RR_matrix)

            # TODO
            # elif NR[J] == 3: # 考虑全部混响(6.21)
            #     VPPR = DONE - USSR
            #     VPPI = -USSI
            #     VSSR = DONE - UPPR
            #     VSSI = -UPPI
            #     VSPR = VPPR * VSSR - VPPI * VSSI - UPSR * USPR + UPSI * USPI
            #     VSPI = VPPI * VSSR + VPPR * VSSI - UPSI * USPR - UPSR * USPI
            #     DETT = VSPR * VSPR + VSPI * VSPI
            #     UPPR = (VPPR * VSPR + VPPI * VSPI) / DETT
            #     UPPI = (VPPI * VSPR - VPPR * VSPI) / DETT
            #     DETR = UPSR
            #     UPSR = (DETR * VSPR + UPSI * VSPI) / DETT
            #     UPSI = (UPSI * VSPR - DETR * VSPI) / DETT
            #     DETR = USPR
            #     USPR = (DETR * VSPR + USPI * VSPI) / DETT
            #     USPI = (USPI * VSPR - DETR * VSPI) / DETT
            #     USSR = (VSSR * VSPR + VSSI * VSPI) / DETT
            #     USSI = (VSSI * VSPR - VSSR * VSPI) / DETT
            #     UHHR = DONE - UHHR
            #     UHHI = UHHI
            #     DETT = UHHR * UHHR + UHHI * UHHI
            #     UHHR = UHHR / DETT
            #     UHHI = UHHI / DETT

            WPPR = UPP * TDPP[LK] + UPS * TDSP[LK]
            # WPPI = UPP * TDPPI(LK) + UPPI * TDPPR(LK) + UPSR * TDSPI(LK) + UPSI * TDSPR(LK)
            WPSR = UPP * TDPS[LK] + UPS * TDSS[LK]
            # WPSI = UPP * TDPSI(LK) + UPPI * TDPSR(LK) + UPSR * TDSSI(LK) + UPSI * TDSSR(LK)
            WSPR = USP * TDPP[LK] + USS * TDSP[LK]
            # WSPI = USP * TDPPI(LK) + USPI * TDPPR(LK) + USSR * TDSPI(LK) + USSI * TDSPR(LK)
            WSSR = USP * TDPS[LK] + USS * TDSS[LK]
            # WSSI = USP * TDPSI(LK) + USPI * TDPSR(LK) + USSR * TDSSI(LK) + USSI * TDSSR(LK)
            WHHR = UHH * TDHH[LK]
            # WHHI = UHH * TDHHI(LK) + UHHI * TDHHR(LK)

            VPPR = RPP * WPPR + RPS * WSPR
            # VPPI = RPPR * WPPI + RPPI * WPPR + RPSR * WSPI + RPSI * WSPR
            VPSR = RPP * WPSR + RPS * WSSR
            # VPSI = RPPR * WPSI + RPPI * WPSR + RPSR * WSSI + RPSI * WSSR
            VSPR = RSP * WPPR + RSS * WSPR
            # VSPI = RSPR * WPPI + RSPI * WPPR + RSSR * WSPI + RSSI * WSPR
            VSSR = RSP * WPSR + RSS * WSSR
            # VSSI = RSPR * WPSI + RSPI * WPSR + RSSR * WSSI + RSSI * WSSR
            VHHR = RHH * WHHR
            # VHHI = RHHR * WHHI + RHHI * WHHR

            RT = 1
            RPP = RDPP[LK] * RT + TUPP[LK] * VPPR + TUPS[LK] * VSPR
            # RPPI = RDPPI(LK) * RT + TUPPR(LK) * VPPI + TUPPI(LK) * VPPR + TUPSR(LK) * VSPI + TUPSI(LK) * VSPR
            RPS = RDPS[LK] * RT + TUPP[LK] * VPSR + TUPS[LK] * VSSR
            # RPSI = RDPSI(LK) * RT + TUPPR(LK) * VPSI + TUPPI(LK) * VPSR + TUPSR(LK) * VSSI + TUPSI(LK) * VSSR
            RSP = RDSP[LK] * RT + TUSS[LK] * VSPR + TUSP[LK] * VPPR
            # RSPI = RDSPI(LK) * RT + TUSSR(LK) * VSPI + TUSSI(LK) * VSPR + TUSPR(LK) * VPPI + TUSPI(LK) * VPPR
            RSS = RDSS[LK] * RT + TUSP[LK] * VPSR + TUSS[LK] * VSSR
            # RSSI = RDSSI(LK) * RT + TUSPR(LK) * VPSI + TUSPI(LK) * VPSR + TUSSR(LK) * VSSI + TUSSI(LK) * VSSR
            RHH = RDHH[LK] * RT + TUHH[LK] * VHHR
            # RHHI = RDHHI(LK) * RT + TUHHR(LK) * VHHI + TUHHI(LK) * VHHR
            J = J - 1

            RD_upper = RD[LK] + TU[LK] @ RD_upper @ RR_matrix @ TD[LK]  # (6.26)
            TD_upper = TD_upper @ RR_matrix @ TD[LK]  # (6.26)
    return RD_upper[0,0]
    # return RPP


respose()
