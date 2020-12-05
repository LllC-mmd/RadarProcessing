import os

import numpy as np
from scipy import stats
import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as plt


'''
############################ Naming Convention ############################
Uppercase for the radar variables in the dB scale
Lowercase for the radar variables in the linear scale
i.e., we have ZH = 10 * log10(Zh) [dBZ]
###########################################################################
'''

def AH_ZH_fitting(csv_data, AH_field="Ah", ZH_field="Zh", img_saved=True):
    df = pd.read_csv(csv_data)
    ah = np.array(df[AH_field])
    ah_lg = np.log10(ah)
    ZH = np.array(df[ZH_field]) / 10.0
    ZH = ZH[:, np.newaxis]

    x_var = np.concatenate((np.ones_like(ZH), ZH), axis=1)
    a, _, _, _ = np.linalg.lstsq(x_var, ah_lg)

    ah_pred = np.power(10.0, a[0]+a[1]*ZH)
    ah_compare = np.concatenate((ah.reshape(1, -1), ah_pred.reshape(1, -1)), axis=0)
    cov = np.cov(ah_compare)
    corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])

    if img_saved:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(ah, ah_pred, s=1, c="navy")
        ax.plot(ah, ah, color="red", zorder=1)
        ax.set_ylabel("$A_{\mathrm{h}}$ (dBZ$\cdot \mathrm{km}^{-1}$) est.", size='large', rotation=90)
        ax.set_xlabel("$K_{\mathrm{DP}}$ (deg$\cdot \mathrm{km}^{-1}$)", size='large')
        ax.set_title("$A_{\mathrm{h}}$=%.4e$\cdot Z_{\mathrm{h}}^{%.4f}$ (CORR=%.4f)" % (10.0**a[0], a[1], corr), fontsize='x-large')
        ax.legend(fontsize='large', loc=2)
        # plt.show()
        plt.savefig("Ah_Zh.png", dpi=400)


def AH_KDP_fitting(csv_data, AH_field="Ah", KDP_field="Kdp", img_saved=True):
    df = pd.read_csv(csv_data)

    AH = np.array(df[AH_field])
    kdp = np.array(df[KDP_field])
    kdp = kdp[:, np.newaxis]
    a, _, _, _ = np.linalg.lstsq(kdp, AH)

    r2 = metrics.r2_score(y_true=AH, y_pred=a[0]*kdp)

    if img_saved:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(kdp, AH, s=1, c="navy")
        ax.plot(kdp, a[0]*kdp, color="red", zorder=1, label="$A_{\mathrm{H}}$=%.4f$\cdot K_{\mathrm{DP}}$" % a[0])
        ax.set_ylabel("$A_{\mathrm{H}}$ (dBZ$\cdot \mathrm{km}^{-1}$)", size='large', rotation=90)
        ax.set_xlabel("$K_{\mathrm{DP}}$ (deg$\cdot \mathrm{km}^{-1}$)", size='large')
        ax.set_title("$A_{\mathrm{H}}$ ~ $K_{\mathrm{DP}}$ ($R^2=$%.4f)" % r2, fontsize='x-large')
        ax.legend(fontsize='large', loc=2)
        # plt.show()
        plt.savefig("AH_KDP.png", dpi=400)


def ADR_KDP_fitting(csv_data, ADR_field="Ahv", KDP_field="Kdp", img_saved=True):
    df = pd.read_csv(csv_data)
    ADR = np.array(df[ADR_field])
    kdp = np.array(df[KDP_field])
    kdp = kdp[:, np.newaxis]
    a, _, _, _ = np.linalg.lstsq(kdp, ADR)

    r2 = metrics.r2_score(y_true=ADR, y_pred=a[0] * kdp)

    if img_saved:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(kdp, ADR, s=1, c="navy")
        ax.plot(kdp, a[0] * kdp, color="red", zorder=1, label="$A_{\mathrm{DR}}$=%.4f$\cdot K_{\mathrm{DP}}$" % a[0])
        ax.set_ylabel("$A_{\mathrm{DR}}$ (dB$\cdot \mathrm{km}^{-1}$)", size='large', rotation=90)
        ax.set_xlabel("$K_{\mathrm{DP}}$ (deg$\cdot \mathrm{km}^{-1}$)", size='large')
        ax.set_title("$A_{\mathrm{DR}}$ ~ $K_{\mathrm{DP}}$ ($R^2=$%.4f)" % r2, fontsize='x-large')
        ax.legend(fontsize='large', loc=2)
        # plt.show()
        plt.savefig("ADR_KDP.png", dpi=400)


# return low bound and high bound coefficients of KDP estimation
# i.e. KDP = C * Zh^alpha * Zdr^beta
# or in its log form, lg(KDP) = lg(C) + alpha * lg(Zh) + beta * lg(Zdr)
# or using ZH, ZDR, lg(KDP) = lg(C) + alpha * ZH/10 + beta * ZDR/10
def KDP_constraints(csv_data, KDP_field="Kdp", ZH_field="Zh", AH_field="Ah", ZDR_field="Zdr", ADR_field="Ahv", img_saved=True):
    df = pd.read_csv(csv_data)
    kdp = np.array(df[KDP_field])
    kdp_lg = np.log10(kdp)
    ZH = np.array(df[ZH_field] + df[AH_field]) / 10.0
    ZH = ZH[:, np.newaxis]
    zdr = np.array(df[ZDR_field] + df[ADR_field]) / 10.0
    zdr = zdr[:, np.newaxis]

    x_var = np.concatenate((np.ones_like(zdr), ZH, zdr), axis=1)
    a, _, _, _ = np.linalg.lstsq(x_var, kdp_lg, rcond=-1)

    kdp_pred = np.power(10, np.dot(x_var, a))
    kdp_compare = np.concatenate((kdp.reshape(1, -1), kdp_pred.reshape(1, -1)), axis=0)
    cov = np.cov(kdp_compare)
    corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])

    if img_saved:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(kdp, kdp_pred, s=1, c="navy")
        ax.plot(kdp, kdp, color="red", zorder=1)
        axis_limit = np.ceil(np.max(kdp))
        ax.set_ylabel("$K_{\mathrm{DP}}$ (deg$\cdot \mathrm{km}^{-1}$) est.", size='large', rotation=90)
        ax.set_ylim(0, axis_limit)
        ax.set_xlabel("$K_{\mathrm{DP}}$ (deg$\cdot \mathrm{km}^{-1}$) intri.", size='large')
        ax.set_xlim(0, axis_limit)
        ax.set_title("$K_{\mathrm{DP}}$ = %.4e$\cdot Z_h^{%.4f}\cdot Z_{dr}^{%.4f}$ (CORR=%.4f)" % (np.power(10, a[0]), a[1], a[2], corr), fontsize='x-large')
        ax.set_aspect('equal')
        # plt.show()
        plt.savefig("KDP_ZDR_ZH.png", dpi=400)
