"""
    ADsurf implementation of the python library disba

    This initial module is not a one-to-one translation from Fortran to Python.
    The code has been adapted and optimized for ADsurf(cpu/gpu inversion).

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
    ..


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

    ca[:,:,0, 0] = cpcq - 2.0 * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy
    ca[:,:,0, 1] = (wvno2 * cpy - cqx) / rho
    ca[:,:,0, 2] = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho
    ca[:,:,0, 3] = (cpz - wvno2 * cqw) / rho
    ca[:,:,0, 4] = -(2.0 * wvno2 * a0pq + xz + wvno2 * wvno2 * wy) / rho2

    ca[:,:,1, 0] = (gmgmk * cpz - gm1sq * cqw) * rho
    ca[:,:,1, 1] = cpcq
    ca[:,:,1, 2] = gammk * cpz - gamm1 * cqw
    ca[:,:,1, 3] = -wz
    ca[:,:,1, 4] = ca[:,:,0, 3]

    ca[:,:,3, 0] = (gm1sq * cpy - gmgmk * cqx) * rho
    ca[:,:,3, 1] = -xy
    ca[:,:,3, 2] = gamm1 * cpy - gammk * cqx
    ca[:,:,3, 3] = ca[:,:,1, 1]
    ca[:,:,3, 4] = ca[:,:,0, 1]

    ca[:,:,4, 0] = (
        -(2.0 * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * xz + gm1sq * gm1sq * wy) * rho2
    )
    ca[:,:,4, 1] = ca[:,:,3, 0]
    ca[:,:,4, 2] = (
        -(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz + gamm1 * gm1sq * wy)
        * rho
    )
    ca[:,:,4, 3] = ca[:,:,1, 0]
    ca[:,:,4, 4] = ca[:,:,0, 0]

    ca[:,:,2, 0] = t * (-(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz + gamm1 * gm1sq * wy)* rho)
    ca[:,:,2, 1] = t * (gamm1 * cpy - gammk * cqx)
    ca[:,:,2, 2] = a0 + 2.0 * (cpcq - ca[:,:,0, 0])
    ca[:,:,2, 3] = t * (gammk * cpz - gamm1 * cqw)
    ca[:,:,2, 4] = t * (-(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho)
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
    w[mask] = dpth
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
    y[mask] = dpth
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

def dltar_matrix(vlist, tlist, d, a, b, rho, ifunc, llw):
    """Select Rayleigh or Love wave period equation."""

    if ifunc == 1:
        return dltar1_matrix(vlist, tlist, d, a, b, rho, llw)
    elif ifunc == 2:
        return dltar4_matrix(vlist, tlist, d, a, b, rho, llw)
    # else:
    #     return fast_delta_matrix(vlist, tlist, d, a, b, rho, llw)

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

def dltar4_matrix(vlist, tlist, d, a, b, rho, llw):
    """
    Rayleigh-wave period equation.
    ----------
    input: 
        vlist: phase velocity list of observation : 1D list
        tlist : period list of observation : 1D list
        d : the thickness : 1D list
        alpha : vp : 1D list
        beta : vs : 1D list
        rho : density : 1D list
        llw : sign for mark contain water layer or not : 0 for contain water layer
    """
    vlist = tensor2numpy(vlist)
    tlist = tensor2numpy(tlist)
    vlist = torch.tensor(vlist).to(torch.float32).reshape(1,-1)
    omega_list = torch.tensor(twopi/tlist).to(torch.float32).reshape(1,-1)
    # Preallocate Dunkin's matrix
    ca = torch.empty((vlist.shape[-1],omega_list.shape[-1],5, 5))
    wvno = (1/vlist.T)*torch.ones((vlist.shape[-1],omega_list.shape[-1])) * omega_list
    omega = torch.ones_like(wvno)*omega_list
    e = torch.zeros((wvno.shape[0],wvno.shape[1],a.shape[-1],5))
    omega = torch.max(omega, torch.tensor(1.0e-4).to(torch.float32))
    wvno2 = wvno * wvno
    xka = omega / a[-1]
    xkb = omega / b[-1]
    wvnop = wvno + xka
    wvnom = torch.abs(wvno - xka)
    ra = torch.sqrt(wvnop * wvnom)
    wvnop = wvno + xkb
    wvnom = torch.abs(wvno - xkb)
    rb = torch.sqrt(wvnop * wvnom)
    t = b[-1] / omega

    # E matrix for the bottom half-space
    gammk = 2.0 * t * t
    gam = gammk * wvno2
    gamm1 = gam - 1.0
    rho1 = rho[-1]
    e[:,:,-1,0] = rho1 * rho1 * (gamm1 * gamm1 - gam * gammk * ra * rb)
    e[:,:,-1,1] = -rho1 * ra
    e[:,:,-1,2] = rho1 * (gamm1 - gammk * ra * rb)
    e[:,:,-1,3] = rho1 * rb
    e[:,:,-1,4] = wvno2 - ra * rb
    # Matrix multiplication from bottom layer upward
    for m in range(len(d) - 2, llw, -1):
        xka = omega / a[m]
        xkb = omega / b[m]
        t = b[m] / omega
        gammk = 2.0 * t * t
        gam = gammk * wvno2
        wvnop = wvno + xka
        wvnom = torch.abs(wvno - xka)
        ra = torch.sqrt(wvnop * wvnom)
        wvnop = wvno + xkb
        wvnom = torch.abs(wvno - xkb)
        rb = torch.sqrt(wvnop * wvnom)

        dpth = d[m]
        rho1 = rho[m]
        p = ra * dpth
        q = rb * dpth

        # Evaluate cosP, cosQ...
        _, _, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz = var_matrix(
            p, q, ra, rb, wvno, xka, xkb, dpth
        )
        
        # Evaluate Dunkin's matrix
        ca = ca[:,:,:,:]
        ca = dnka_matrix(
            wvno2, gam, gammk, rho1, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz, ca
        ) #[m,n,5,5]
        
        # multi layer 
        ee = torch.zeros(ca.shape[0],ca.shape[1],1,5) # [m,n,1,5]
        e_temp = e[:,:,[m+1],:] # [m,n,1,5]
        ee = torch.matmul(e_temp,ca) # [m,n,1,5] * [m,n,5,5] = [m,n,1,5]
        # ee,_ = normc_matrix(ee)
        e[:,:,m,:] = ee.squeeze()[:,:,:]
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
        dlt = e[:,:,0,0]

    return dlt