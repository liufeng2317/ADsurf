"""
Numba implementation of the Fortran program surf96.

This module is not a one-to-one translation from Fortran to Python.
The code has been adapted and optimized for Numba.

..

    COMPUTER PROGRAMS IN SEISMOLOGY
    VOLUME IV

    COPYRIGHT 1986, 1991
    D. R. Russell, R. B. Herrmann
    Department of Earth and Atmospheric Sciences
    Saint Louis University
    221 North Grand Boulevard
    St. Louis, Missouri 63103
    U. S. A.

"""
import numpy as np
import torch
from ADsurf._utils import numpy2tensor,tensor2numpy
torch.set_printoptions(precision=8)

__all__ = [
    "surf96",
]

twopi = 2.0 * np.pi

class DispersionError(Exception):
    pass


def normc_matrix(ee):
    """Normalize Haskell or Dunkin vectors."""
    t1 = torch.zeros((ee.shape[0],ee.shape[1],ee.shape[2]))
    t1 = torch.max(torch.abs(ee),dim=3,keepdim=True)[0]
    mask = t1<1.0e-40
    t1[mask] = 1.0
    ee = ee/t1
    ex = torch.log(t1)
    return ee, ex

def dnka_matrix(wvno2, gam, gammk, rho, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz, ca):
    """Dunkin's matrix."""
    gamm1 = gam - 1.0
    twgm1 = gam + gamm1
    gmgmk = gam * gammk
    gmgm1 = gam * gamm1
    gm1sq = gamm1 * gamm1

    rho2 = rho * rho
    a0pq = a0 - cpcq
    t = -2.0 * wvno2
    ca = torch.zeros_like(ca)

    ca[:,:,:,0, 0] = cpcq - 2.0 * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy
    ca[:,:,:,0, 1] = (wvno2 * cpy - cqx) / rho
    ca[:,:,:,0, 2] = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho
    ca[:,:,:,0, 3] = (cpz - wvno2 * cqw) / rho
    ca[:,:,:,0, 4] = -(2.0 * wvno2 * a0pq + xz + wvno2 * wvno2 * wy) / rho2

    ca[:,:,:,1, 0] = (gmgmk * cpz - gm1sq * cqw) * rho
    ca[:,:,:,1, 1] = cpcq
    ca[:,:,:,1, 2] = gammk * cpz - gamm1 * cqw
    ca[:,:,:,1, 3] = -wz
    ca[:,:,:,1, 4] = ca[:,:,:,0, 3]

    ca[:,:,:,3, 0] = (gm1sq * cpy - gmgmk * cqx) * rho
    ca[:,:,:,3, 1] = -xy
    ca[:,:,:,3, 2] = gamm1 * cpy - gammk * cqx
    ca[:,:,:,3, 3] = ca[:,:,:,1, 1]
    ca[:,:,:,3, 4] = ca[:,:,:,0, 1]

    ca[:,:,:,4, 0] = (
        -(2.0 * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * xz + gm1sq * gm1sq * wy) * rho2
    )
    ca[:,:,:,4, 1] = ca[:,:,:,3, 0]
    ca[:,:,:,4, 2] = (
        -(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz + gamm1 * gm1sq * wy)
        * rho
    )
    ca[:,:,:,4, 3] = ca[:,:,:,1, 0]
    ca[:,:,:,4, 4] = ca[:,:,:,0, 0]

    ca[:,:,:,2, 0] = t * (-(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz + gamm1 * gm1sq * wy)* rho)
    ca[:,:,:,2, 1] = t * (gamm1 * cpy - gammk * cqx)
    ca[:,:,:,2, 2] = a0 + 2.0 * (cpcq - ca[:,:,:,0, 0])
    ca[:,:,:,2, 3] = t * (gammk * cpz - gamm1 * cqw)
    ca[:,:,:,2, 4] = t * (-(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho)
    return ca

def var_matrix(p, q, ra, rb, wvno, xka, xkb, dpth):
    """Find variables cosP, cosQ, sinP, sinQ..."""
    # Examine P-wave eigenfunctions
    # Checking whether c > vp, c = vp or c < vp
    pex = torch.zeros_like(wvno)
    fac = torch.zeros_like(wvno)
    sinp = torch.zeros_like(wvno)
    w = torch.zeros_like(wvno)
    x = torch.zeros_like(wvno)
    cosp = torch.zeros_like(wvno)
    # wvno < xka
    mask = wvno<xka
    sinp[mask] = torch.sin(p[mask])
    w[mask] = sinp[mask]/ra[mask]
    x[mask] = -ra[mask] * sinp[mask]
    cosp[mask] = torch.cos(p[mask])
    
    # wvno > xka
    mask = wvno>xka
    pex[mask] = p[mask]
    mask1 = (wvno>xka) & (p<16.0)
    fac[mask1] = torch.exp(-2.0 * p[mask1])
    cosp[mask] = (1.0 + fac[mask]) * 0.5
    sinp[mask] = (1.0 - fac[mask]) * 0.5
    w[mask] = sinp[mask]/ra[mask]
    x[mask] = ra[mask]*sinp[mask]

    # wvno = xka
    mask = wvno==xka
    cosp[mask] = 1.0
    w[mask] = dpth[mask]
    x[mask] = 0.0

    # Examine S-wave eigenfunctions
    # Checking whether c > vs, c = vs or c < vs
    sex = torch.zeros_like(wvno)
    fac = torch.zeros_like(wvno)
    sinq = torch.zeros_like(wvno)
    y = torch.zeros_like(wvno)
    z = torch.zeros_like(wvno)
    cosq = torch.zeros_like(wvno)
    # wvno < xkb
    mask = wvno<xkb
    sinq[mask] = torch.sin(q[mask])
    y[mask] = sinq[mask]/rb[mask]
    z[mask] = -rb[mask] * sinq[mask]
    cosq[mask] = torch.cos(q[mask])
    
    # wvno > xkb
    mask = wvno>xkb
    sex[mask] = q[mask]
    mask1 = (wvno>xkb)&(q<16.0)# 同时慢著wvno>xkb和q<16.0的条件
    fac[mask1] = torch.exp(-2.0 * q[mask1]) 
    cosq[mask] = (1.0 + fac[mask]) * 0.5
    sinq[mask] = (1.0 - fac[mask]) * 0.5
    y[mask] = sinq[mask]/rb[mask]
    z[mask] = rb[mask]*sinq[mask]

    # wvno = xkb
    mask = wvno==xkb
    cosq[mask] = 1.0
    y[mask] = dpth[mask]
    z[mask] = 0.0


    # Form eigenfunction products for use with compound matrices
    a0 = torch.zeros_like(wvno)
    exa = pex + sex
    a0[exa<60.0] = torch.exp(-exa[exa<60.0])
    cpcq = cosp * cosq
    cpy = cosp * y
    cpz = cosp * z
    cqw = cosq * w
    cqx = cosq * x
    xy = x * y
    xz = x * z
    wy = w * y
    wz = w * z

    # fac = torch.zeros_like(wvno)
    # qmp = sex - pex
    # fac[qmp>-40.0] = torch.exp(qmp[qmp>-40.0])
    # cosq *= fac
    # y *= fac
    # z *= fac

    return w, cosp, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz

def dltar_matrix(vlist, tlist, d, a, b, rho, ifunc, llw,device="cpu"):
    """Select Rayleigh or Love wave period equation."""

    if ifunc == 1:
        return dltar1_matrix(vlist, tlist, d, a, b, rho, llw,device)
    elif ifunc == 2:
        return dltar4_matrix(vlist, tlist, d, a, b, rho, llw,device)
    else:
        return fast_delta_matrix(vlist, tlist, d, a, b, rho, llw,device)

def dltar1_matrix(wvno, omega, d, a, b, rho, llw):
    """Love-wave period equation."""
    beta1 = b[-1]
    rho1 = rho[-1]
    xkb = omega / beta1
    wvnop = wvno + xkb
    wvnom = np.abs(wvno - xkb)
    rb = np.sqrt(wvnop * wvnom)
    e1 = rho1 * rb
    e2 = 1.0 / (beta1 * beta1)

    for m in range(len(d) - 2, llw, -1):
        beta1 = b[m]
        rho1 = rho[m]
        xmu = rho1 * beta1 * beta1
        xkb = omega / beta1
        wvnop = wvno + xkb
        wvnom = np.abs(wvno - xkb)
        rb = np.sqrt(wvnop * wvnom)
        q = d[m] * rb

        if wvno < xkb:
            sinq = np.sin(q)
            y = sinq / rb
            z = -rb * sinq
            cosq = np.cos(q)
        elif wvno == xkb:
            cosq = 1.0
            y = d[m]
            z = 0.0
        else:
            fac = np.exp(-2.0 * q) if q < 16.0 else 0.0
            cosq = (1.0 + fac) * 0.5
            sinq = (1.0 - fac) * 0.5
            y = sinq / rb
            z = rb * sinq

        e10 = e1 * cosq + e2 * xmu * z
        e20 = e1 * y / xmu + e2 * cosq
        xnor = np.abs(e10)
        ynor = np.abs(e20)
        xnor = max(xnor, ynor)
        if xnor < 1.0e-40:
            xnor = 1.0
        e1 = e10 / xnor
        e2 = e20 / xnor

    return e1

def dltar4_matrix(vlist, tlist, d, a, b, rho, llw,device):
    """
    Rayleigh-wave period equation.
    ----------
    input: 
        vlist: phase velocity list of observation : 2D list [station,vc]
        tlist : period list of observation : 2D list [station,t]
        d : the thickness : 2D list [station,d]
        alpha : vp : 2D list [station,alpha]
        beta : vs : 2D list [station,beta]
        rho : density : 2D list [station,rho]
        llw : sign for mark contain water layer or not : 0 for contain water layer
    """
    vlist = numpy2tensor(vlist).reshape(vlist.shape[0],vlist.shape[-1],1)# [station,vc] ==> [station,1,vc]
    tlist = numpy2tensor(twopi/tlist).reshape(tlist.shape[0],1,tlist.shape[-1])# [station,t] ==> [station,1,t]
    tlist = numpy2tensor(np.repeat(tlist.cpu().detach(),vlist.shape[1],1)).to(device) # [station,clist,t]
    vlist = numpy2tensor(np.repeat(vlist.cpu().detach(),tlist.shape[-1],-1)).to(device) #[station,clist,t]
    omega = tlist.to(device)
    wvno = (1/vlist*omega)
    # Preallocate Dunkin's matrix [station,vc,t,5,5]
    ca = torch.empty((omega.shape[0],omega.shape[1],omega.shape[2],5, 5)).to(device)

    # Dukin's temp matrix [station,vc,t,5]
    e = torch.zeros((omega.shape[0],omega.shape[1],omega.shape[2],a.shape[-1],5)).to(device)
    omega = torch.max(omega, torch.tensor(1.0e-4).to(torch.float32))
    wvno2 = wvno * wvno
    xka = omega / a[:,-1].reshape(-1,1,1)
    xkb = omega / b[:,-1].reshape(-1,1,1)
    wvnop = wvno + xka
    wvnom = torch.abs(wvno - xka)
    ra = torch.sqrt(wvnop * wvnom)
    wvnop = wvno + xkb
    wvnom = torch.abs(wvno - xkb)
    rb = torch.sqrt(wvnop * wvnom)
    t = b[:,-1].reshape(-1,1,1) / omega

    # E matrix for the bottom half-space
    gammk = 2.0 * t * t
    gam = gammk * wvno2
    gamm1 = gam - 1.0
    rho1 = rho[:,-1].reshape(-1,1,1)
    e[:,:,:,-1,0] = rho1 * rho1 * (gamm1 * gamm1 - gam * gammk * ra * rb)
    e[:,:,:,-1,1] = -rho1 * ra
    e[:,:,:,-1,2] = rho1 * (gamm1 - gammk * ra * rb)
    e[:,:,:,-1,3] = rho1 * rb
    e[:,:,:,-1,4] = wvno2 - ra * rb

    # Matrix multiplication from bottom layer upward
    for m in range(d.shape[-1] - 2, llw, -1):
        xka = omega / a[:,m].reshape(-1,1,1)
        xkb = omega / b[:,m].reshape(-1,1,1)
        t = b[:,m].reshape(-1,1,1) / omega
        gammk = 2.0 * t * t
        gam = gammk * wvno2
        wvnop = wvno + xka
        wvnom = torch.abs(wvno - xka)
        ra = torch.sqrt(wvnop * wvnom)
        wvnop = wvno + xkb
        wvnom = torch.abs(wvno - xkb)
        rb = torch.sqrt(wvnop * wvnom)

        dpth = torch.ones_like(wvno)*d[:,m].reshape(-1,1,1)
        rho1 = rho[:,m].reshape(-1,1,1)
        p = ra * dpth
        q = rb * dpth

        # Evaluate cosP, cosQ...
        _, _, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz = var_matrix(
            p, q, ra, rb, wvno, xka, xkb, dpth
        )
        # Evaluate Dunkin's matrix
        ca = ca[:,:,:,:,:]
        ca = dnka_matrix(
            wvno2, gam, gammk, rho1, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz, ca
        ) #[m,n,5,5]
        # multi layer 
        ee = torch.zeros(ca.shape[0],ca.shape[1],ca.shape[2],1,5).to(device) # [m,n,1,5]
        e_temp = e[:,:,:,[m+1],:] # [m,n,1,5]
        ee = torch.matmul(e_temp,ca) # [m,n,1,5] * [m,n,5,5] = [m,n,1,5]
        # ee,_ = normc_matrix(ee)
        e[:,:,:,m,:] = ee.squeeze()[:,:,:,:]
    
    if llw == 0:
        xka = omega / a[0]
        wvnop = wvno + xka
        wvnom = torch.abs(wvno - xka)
        ra = torch.sqrt(wvnop * wvnom)
        dpth = d[0]
        rho1 = rho[0]
        p = ra * dpth
        w, cosp, _, _, _, _, _, _, _, _, _, _ = var_matrix(
            p, q, ra, 1.0e-5, wvno, xka, xkb, dpth
        )
        dlt = cosp * e[:,:,0] - rho1 * w * e[:,:,1]
    else:
        dlt = e[:,:,:,0,0]
        # 修改输出
        # det_A = e[:,:,0,0]
        # det_B = torch.sign(det_A)
        # det_C = torch.ones_like(det_A)
        # det_C[:-1,:] = det_B[:-1,:]+det_B[1:,:]
        # det_D = -torch.sign(torch.abs(det_C) - 2)
        # det_E = det_D * vlist.reshape(-1,1)
        # det_F = det_E[:,:]
        # vlist_new = vlist.reshape(-1)
        # for i in range(det_F.shape[1]):
        #     mask = det_F[:,i]>0
        #     if mask.any():
        #         det_F[:,i] = torch.where(det_F[:,i]>vlist_new[mask][0],torch.zeros_like(det_F[:,i]),det_F[:,i])
        # dlt = det_F

    return dlt

def fast_delta_matrix(wvno, omega, d, alpha, beta, rho, llw):
    """
    Fast delta matrix.

    After Buchen and Ben-Hador (1996).

    """
    mmax = len(d)

    # Handle water layer
    if llw == 0:
        beta = beta * 1.0
        beta[0] = 1.0e-8

    # Phase velocity
    c = omega / wvno
    c2 = c * c

    # Rayleigh-wave fast delta matrix
    gam0 = beta[0] ** 2.0 / c2
    mu0 = rho[0] * beta[0] ** 2.0
    t0 = 2.0 - c2 / beta[0] ** 2.0

    X = np.zeros(5, dtype=np.complex_)
    X[0] = 2.0 * t0
    X[1] = -t0 * t0
    X[4] = -4.0
    X *= mu0 * mu0
    X, _ = normc(X, 5)

    for i in range(mmax - 1):
        scale = 0
        gam1 = beta[i + 1] ** 2.0 / c2
        eps = rho[i + 1] / rho[i]
        eta = 2.0 * (gam0 - eps * gam1)
        a = eps + eta
        ap = a - 1.0
        b = 1.0 - eta
        bp = b - 1.0
        gam0 = gam1

        r = (
            np.sqrt(1.0 - c2 / alpha[i] ** 2.0)
            if c < alpha[i]
            else np.sqrt(c2 / alpha[i] ** 2.0 - 1.0) * 1j
            if c > alpha[i]
            else 0.0
        )
        s = (
            np.sqrt(1.0 - c2 / beta[i] ** 2.0)
            if c < beta[i]
            else np.sqrt(c2 / beta[i] ** 2.0 - 1.0) * 1j
            if c > beta[i]
            else 0.0
        )

        if c < alpha[i]:
            Ca = np.cosh(wvno * r * d[i])
            Sa = np.sinh(wvno * r * d[i])
        elif c > alpha[i]:
            Ca = np.cos(wvno * r.imag * d[i])
            Sa = np.sin(wvno * r.imag * d[i]) * 1j
        else:
            Ca = 1.0
            Sa = 0.0

        if c < beta[i]:
            Cb = np.cosh(wvno * s * d[i])
            Sb = np.sinh(wvno * s * d[i])

            # Handle hyperbolic overflow
            if wvno * s.real * d[i] > 80.0:
                scale = 1
                Cb /= Ca
                Sb /= Ca

        elif c > beta[i]:
            Cb = np.cos(wvno * s.imag * d[i])
            Sb = np.sin(wvno * s.imag * d[i]) * 1j

        else:
            Cb = 1.0
            Sb = 0.0

        if i == 0 and llw == 0:
            Cb = 1.0
            Sb = 0.0

        p1 = Cb * X[1] + s * Sb * X[2]
        p2 = Cb * X[3] + s * Sb * X[4]
        p3 = Cb * X[2]
        p4 = Cb * X[4]
        if c == beta[i]:
            p3 = p3 + wvno * d[i] * X[1]
            p4 = p4 + wvno * d[i] * X[3]
        else:
            p3 = p3 + Sb * X[1] / s
            p4 = p4 + Sb * X[3] / s

        if scale == 1:
            q1 = p1 - r * p2
            q3 = p3 - r * p4
            q2 = p4
            q4 = p2
            if c != alpha[i]:
                q2 -= p3 / r
                q4 -= p1 / r
        else:
            q1 = Ca * p1 - r * Sa * p2
            q3 = Ca * p3 - r * Sa * p4
            q2 = Ca * p4
            q4 = Ca * p2
            if c == alpha[i]:
                q2 -= wvno * d[i] * p3
                q4 -= wvno * d[i] * p1
            else:
                q2 -= Sa * p3 / r
                q4 -= Sa * p1 / r

        y1 = a * q1
        y2 = ap * q2
        z1 = bp * q1
        z2 = b * q2
        if scale == 0:
            y1 = y1 + ap * X[0]
            y2 = y2 + a * X[0]
            z1 = z1 + b * X[0]
            z2 = z2 + bp * X[0]

        X[0] = bp * y1 + b * y2
        X[1] = a * y1 + ap * y2
        X[2] = eps * q3
        X[3] = eps * q4
        X[4] = bp * z1 + b * z2
        X, _ = normc(X, 5)

    r = (
        np.sqrt(1.0 - c2 / alpha[-1] ** 2)
        if c < alpha[-1]
        else np.sqrt(c2 / alpha[-1] ** 2 - 1.0) * 1j
        if c > alpha[-1]
        else 0.0
    )
    s = (
        np.sqrt(1.0 - c2 / beta[-1] ** 2)
        if c < beta[-1]
        else np.sqrt(c2 / beta[-1] ** 2 - 1.0) * 1j
        if c > beta[-1]
        else 0.0
    )

    return np.real(X[1] + s * X[3] - r * (X[3] + s * X[4]))

def nevill(t, c1, c2, del1, del2, d, a, b, rho, ifunc, llw, ca):
    """Hybrid method for refining root once it has been bracketted."""
    x = np.zeros(20, dtype=np.float64)
    y = np.zeros(20, dtype=np.float64)

    # Initial guess
    omega = twopi / t
    c3 = 0.5 * (c1 + c2)
    del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)
    nev = 1
    nctrl = 1

    while True:
        nctrl = nctrl + 1
        if nctrl >= 100:
            break

        # Make sure new estimate is inside the previous values
        # If not, perform interval halving
        if c3 < min(c1, c2) or c3 > max(c1, c2):
            nev = 0
            c3 = 0.5 * (c1 + c2)
            del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)

        s13 = del1 - del3
        s32 = del3 - del2

        # Define new bounds according to the sign of the period equation
        if np.sign(del3) * np.sign(del1) < 0.0:
            c2 = c3
            del2 = del3
        else:
            c1 = c3
            del1 = del3

        # Check for convergence
        if np.abs(c1 - c2) <= 1.0e-6 * c1:
            break

        # If the slopes are not the same between c1, c2 and c3
        # Do not use Neville iteration
        if np.sign(s13) != np.sign(s32):
            nev = 0

        # If the period equation differs by more than a factor of 10
        # Use interval halving to avoid poor behavior of polynomial fit
        ss1 = np.abs(del1)
        s1 = 0.01 * ss1
        ss2 = np.abs(del2)
        s2 = 0.01 * ss2
        if s1 > ss2 or s2 > ss1 or nev == 0:
            c3 = 0.5 * (c1 + c2)
            del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)
            nev = 1
            m = 1
        else:
            if nev == 2:
                x[m - 1] = c3
                y[m - 1] = del3
            else:
                x[0] = c1
                y[0] = del1
                x[1] = c2
                y[1] = del2
                m = 1

            # Perform Neville iteration
            flag = 1
            for kk in range(m):
                j = m - kk
                denom = y[m] - y[j]
                if np.abs(denom) < 1.0e-10 * np.abs(y[m]):
                    flag = 0
                    break
                else:
                    x[j - 1] = (-y[j - 1] * x[j] + y[m] * x[j - 1]) / denom

            if flag:
                c3 = x[0]
                del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)
                nev = 2
                m = m + 1
                m = min(m, 10)
            else:
                c3 = 0.5 * (c1 + c2)
                del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)
                nev = 1
                m = 1

    return c3

def getsol(t1, c1, clow, dc, cm, betmx, ifirst, del1st, d, a, b, rho, ifunc, llw, ca):
    """Bracket dispersion curve and then refine it."""
    # Bracket solution
    omega = twopi / t1
    del1 = dltar(omega / c1, omega, d, a, b, rho, ifunc, llw, ca)
    del1st = del1 if ifirst else del1st
    idir = -1.0 if not ifirst and np.sign(del1st) * np.sign(del1) < 0.0 else 1.0

    # idir indicates the direction of the search for the true phase velocity from the initial estimate
    while True:
        c2 = c1 + idir * dc

        if c2 <= clow:
            idir = 1.0
            c1 = clow
        else:
            omega = twopi / t1
            del2 = dltar(omega / c2, omega, d, a, b, rho, ifunc, llw, ca)

            if np.sign(del1) != np.sign(del2):
                c1 = nevill(t1, c1, c2, del1, del2, d, a, b, rho, ifunc, llw, ca)
                iret = c1 > betmx
                break

            c1 = c2
            del1 = del2

            iret = c1 < cm or c1 >= betmx + dc
            if iret:
                break

    return c1, del1st, iret

def gtsolh(a, b):
    """Starting solution."""
    c = 0.95 * b

    for _ in range(5):
        gamma = b / a
        kappa = c / b
        k2 = kappa ** 2
        gk2 = (gamma * kappa) ** 2
        fac1 = np.sqrt(1.0 - gk2)
        fac2 = np.sqrt(1.0 - k2)
        fr = (2.0 - k2) ** 2 - 4.0 * fac1 * fac2
        frp = -4.0 * (2.0 - k2) * kappa
        frp = frp + 4.0 * fac2 * gamma * gamma * kappa / fac1
        frp = frp + 4.0 * fac1 * kappa / fac2
        frp /= b
        c = c -  fr / frp

    return c

def getc(t, d, a, b, rho, mode, ifunc, dc):
    """Get phase velocity dispersion curve."""
    # Initialize arrays
    kmax = len(t)
    c = torch.zeros(kmax)
    cg = torch.zeros(kmax)

    # Preallocate Dunkin's matrix
    ca = torch.empty((5, 5))

    # Check for water layer
    llw = 0 if b[0] <= 0.0 else -1

    # Find the extremal velocities to assist in starting search
    betmx = -1.0e20
    betmn = 1.0e20
    mmax = len(d)
    for i in range(mmax):
        if b[i] > 0.01 and b[i] < betmn: # vs over than 0.01 and less than betmn
            betmn = b[i]
            jmn = i
            jsol = False
        elif b[i] < 0.01 and a[i] < betmn: # vs less than 0.01 and vp less than betmn
            betmn = a[i]
            jmn = i
            jsol = True
        betmx = max(betmx, b[i])

    # Solid layer solve halfspace period equation
    cc = betmn if jsol else gtsolh(a[jmn], b[jmn])

    # Back off a bit to get a starting value at a lower phase velocity
    cc *= 0.9
    c1 = cc
    cm = cc

    one = 1.0e-2
    onea = 1.5

    for iq in range(mode + 1):
        ibeg = 0
        iend = kmax

        del1st = 0.0
        for k in range(ibeg, iend):
            # Get initial phase velocity estimate to begin search
            ifirst = k == ibeg
            if ifirst and iq == 0:
                clow = cc
                c1 = cc
            elif ifirst and iq > 0:
                clow = c1
                c1 = c[ibeg] + one * dc
            elif not ifirst and iq > 0:
                clow = c[k] + one * dc
                c1 = max(c[k - 1], clow)
            elif not ifirst and iq == 0:
                clow = cm
                c1 = c[k - 1] - onea * dc
            # Bracket root and refine it
            c1, del1st, iret = getsol(
                t[k],
                c1,
                clow,
                dc,
                cm,
                betmx,
                ifirst,
                del1st,
                d,
                a,
                b,
                rho,
                ifunc,
                llw,
                ca,
            )

            if iret:
                if iq > 0:
                    for i in range(k, kmax):
                        cg[i] = 0.0

                    if iq == mode:
                        return cg
                    else:
                        c1 = 0.0
                        break
                else:
                    raise DispersionError("failed to find root for fundamental mode")

            c[k] = c1
            cg[k] = c[k]
            c1 = 0.0
    
    ##############################################################
    #  LIUFENG TEST
    # c_list = torch.arange(0.5,0.9,dc,requires_grad=True)
    # vnum = len(c_list)
    # det_A = torch.zeros((vnum,kmax))
    # for i in range(kmax): # frequency
    #     omega = twopi / t[i]
    #     for j in range(vnum): # velocity
    #         del1 = dltar(omega / c_list[j], omega, d, a, b, rho, ifunc, llw, ca)
    #         det_A[j,i] = del1
    # det_B = torch.roll(det_A,-1,0)
    # det_B[-1,:] = torch.zeros_like(det_B[-1,:])
    # det_C = F.relu(-torch.sign(det_A*det_B))
    # cg_test = torch.zeros_like(cg)
    # for idx in range(kmax):
    #     G_ind = torch.where(det_C[idx]==1)[0]
    #     if len(G_ind) > 0:
    #         cg_test[idx] = c_list[G_ind[0]]
    # loss = torch.sum((cg - cg_test)**2/len(cg))
    # loss.backward(retain_graph=True)
    # X,Y = np.meshgrid(1/t,c_list.detach().numpy())
    # plt.figure(figsize=(12,8))
    # plt.contourf(X,Y,det_A.detach().numpy())
    # plt.colorbar()
    # for x,y in zip(1/t,cg):
    #     plt.plot(x,y,'o',markersize=10)
    # plt.show()
    ################################################################

    return cg

def surf96(t, d, a, b, rho, mode=0, itype=0, ifunc=2, dc=0.005, dt=0.025):
    """
    Get phase or group velocity dispersion curve.

    Parameters
    ----------
    t : array_like
        Periods (in s).
    d : array_like
        Layer thickness (in km).
    a : array_like
        Layer P-wave velocity (in km/s).
    b : array_like
        Layer S-wave velocity (in km/s).
    rho : array_like
        Layer density (in g/cm3).
    mode : int, optional, default 0
        Mode number (0 if fundamental).
    itype : int, optional, default 0
        Velocity type:
            - 0: phase velocity,
            - 1: group velocity.
    ifunc : int, optional, default 2
        Select wave type and algorithm for period equation:
            - 1: Love-wave (Thomson-Haskell method),
            - 2: Rayleigh-wave (Dunkin's matrix),
            - 3: Rayleigh-wave (fast delta matrix).
    dc : scalar, optional, default 0.005
        Phase velocity increment for root finding.
    dt : scalar, optional, default 0.025
        Frequency increment (%) for calculating group velocity.

    Returns
    -------
    array_like
        Phase or group dispersion velocity.

    """
    nt = len(t)
    t1 = torch.empty(nt) # a new time list

    if itype == 1:
        fac = 1.0 + dt
        for i in range(nt):
            t1[i] = t[i] / fac
    else:
        for i in range(nt):
            t1[i] = t[i]

    c1 = getc(t1, d, a, b, rho, mode, ifunc, dc)

    if itype == 1:
        t2 = torch.empty(nt)
        fac = 1.0 - dt
        for i in range(nt):
            t2[i] = t[i] / fac

        c2 = getc(t2, d, a, b, rho, mode, ifunc, dc)

        for i in range(nt):
            if c2[i] > 0.0:
                t1i = 1.0 / t1[i]
                t2i = 1.0 / t2[i]
                c1[i] = (t1i - t2i) / (t1i / c1[i] - t2i / c2[i])
            else:
                c1[i] = 0.0

    return c1
