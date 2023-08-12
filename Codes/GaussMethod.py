import numpy as np
from astropy import constants as cts
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
#import poliastro as pa
import matplotlib.pyplot as plt

au = cts.au.to(u.Unit('km')).value
mu_Sun = cts.GM_sun.to(u.Unit('au3 / day2')).value
c_light = cts.c.to(u.Unit('au/day'))

def observation(string):
    hdul = fits.open(string)
    time = hdul[0].header["DATE-OBS"]
    observatory = hdul[0].header["OBSERVAT"]
    observatory_lat = hdul[0].header["SITELAT"]
    observatory_long = hdul[0].header["SITELONG"]
    location = (observatory_lat, observatory_long)
    t = Time(time, format='isot', scale='utc')
    return t, location, observatory 

def c3(z):
    """Compute the Stumpff function.

       Args:
           z

       Returns:
           the value of the Stumpff function
    """
    if z > 0:
        rz = np.sqrt(z)
        return ((rz - np.sin(rz)) / rz**3)
    elif z < 0:
        rz = np.sqrt(-z)
        return ((np.sinh(rz)-rz)/rz**3)
    elif z == 0:
        return 1/6
    
def c2(z):
    """Compute the Stumpff function.

       Args:
           z

       Returns:
           the value of the Stumpff function
    """
    if z > 0:
        rz = np.sqrt(z)
        return ((1 - np.cos(rz)) / z)
    elif z < 0:
        rz = np.sqrt(-z)
        return ((np.cosh(rz)-1)/(-z))
    elif z == 0:
        return 1/2

def unit_vec(ra, dec):
    """Compute line-of-sight (LOS) vector for given values of right ascension
    and declination. Both angles must be provided in radians.

       Args:
           ra_rad (float): right ascension (rad)
           dec_rad (float): declination (rad)

       Returns:
           1x3 numpy array: cartesian components of LOS vector.
    """
    cosa_cosd = np.cos(ra)*np.cos(dec)
    sina_cosd = np.sin(ra)*np.cos(dec)
    sind = np.sin(dec)
    return np.array((cosa_cosd, sina_cosd, sind))

def f_series(mu, r2, tau):
    """Compute 1st order approximation to Lagrange's f function.

       Args:
           mu (float): gravitational parameter attracting body
           r2 (float): radial distance
           tau (float): time interval

       Returns:
           float: Lagrange's f function value
    """
    return 1.0-0.5*(mu/(r2**3))*(tau**2)

def g_series(mu, r2, tau):
    """Compute 1st order approximation to Lagrange's g function.

       Args:
           mu (float): gravitational parameter attracting body
           r2 (float): radial distance
           tau (float): time interval

       Returns:
           float: Lagrange's g function value
    """
    return tau-(1.0/6.0)*(mu/(r2**3))*(tau**3)

def lagrangef_(xi, z, r):
    """Compute current value of Lagrange's f function.

       Args:
           xi (float): universal Kepler anomaly
           z (float): xi**2/alpha
           r (float): radial distance

       Returns:
           float: Lagrange's f function value
    """
    return 1.0-(xi**2)*c2(z)/r

def lagrangeg_(tau, xi, z, mu):
    """Compute current value of Lagrange's g function.

       Args:
           tau (float): time interval
           xi (float): universal Kepler anomaly
           z (float): xi**2/alpha
           r (float): radial distance

       Returns:
           float: Lagrange's g function value
    """
    return tau-(xi**3)*c3(z)/np.sqrt(mu)

def gauss_velocity(r1, r2, r3, t1, t2, t3, r2_star=None):
    """Calculate the velocity vector given the position vectors and 
    the observations times.

       Args:
           r1: position of the asteroid at the first observation
           r2: position of the asteroid at the second observation
           r3: position of the asteroid at the third observation
           t1: time of the first observation
           t2: time of the second observation
           t3: time of the third observation

       Returns:
           return the velocity vector at the second observation
    """    

    mu = mu_Sun

    tau1 = (t1 - t2).jd
    tau3 = (t3 - t2).jd
    #tau = (tau3 - tau1)

    if r2_star == None:
        r2_star = np.linalg.norm(r2)

    f1 = f_series(mu, r2_star, tau1)
    f3 = f_series(mu, r2_star, tau3)

    g1 = g_series(mu, r2_star, tau1)
    g3 = g_series(mu, r2_star, tau3)

    v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1)

    return v2

def univkepler(dt, x, y, z, u, v, w, mu, iters=5, atol=1e-15):
    """Compute the current value of the universal Kepler anomaly, xi.

       Args:
           dt (float): time interval
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity
           mu (float): gravitational parameter
           iters (int): number of iterations of Newton-Raphson process
           atol (float): absolute tolerance of Newton-Raphson process

       Returns:
           float: alpha = 1/a
    """
    # compute preliminaries
    r0 = np.sqrt((x**2)+(y**2)+(z**2))
    v20 = (u**2)+(v**2)+(w**2)
    vr0 = (x*u+y*v+z*w)/r0
    alpha0 = (2.0/r0)-(v20/mu)
    # compute initial estimate for xi
    xi = np.sqrt(mu)*np.abs(alpha0)*dt
    i = 0
    ratio_i = 1.0
    while np.abs(ratio_i)>atol and i<iters:
        xi2 = xi**2
        z_i = alpha0*(xi2)
        a_i = (r0*vr0)/np.sqrt(mu)
        b_i = 1.0-alpha0*r0

        C_z_i = c2(z_i)
        S_z_i = c3(z_i)

        if np.isinf(C_z_i) == True or np.isinf(S_z_i) == True:
            return np.nan

        f_i = a_i*xi2*C_z_i + b_i*(xi**3)*S_z_i + r0*xi - np.sqrt(mu)*dt
        g_i = a_i*xi*(1.0-z_i*S_z_i) + b_i*xi2*C_z_i+r0
        ratio_i = f_i/g_i
        xi = xi - ratio_i
        i += 1

    return xi

def gauss_method(obs_radec, obs_t, R, mu, r2_root_ind=0):
    """Perform core Gauss method.

       Args:
           obs_radec (1x3 SkyCoord array): three rad/dec observations
           obs_t (1x3 array): three times of observations
           R (1x3 array): three observer position vectors
           mu (float): gravitational parameter of center of attraction
           r2_root_ind (int): index of Gauss polynomial root

       Returns:
           r1 (1x3 array): estimated position at first observation
           r2 (1x3 array): estimated position at second observation
           r3 (1x3 array): estimated position at third observation
           v2 (1x3 array): estimated velocity at second observation
           D (3x3 array): auxiliary matrix
           rho1 (1x3 array): LOS vector at first observation
           rho2 (1x3 array): LOS vector at second observation
           rho3 (1x3 array): LOS vector at third observation
           tau1 (float): time interval from second to first observation
           tau3 (float): time interval from second to third observation
           f1 (float): estimated Lagrange's f function value at first observation
           g1 (float): estimated Lagrange's g function value at first observation
           f3 (float): estimated Lagrange's f function value at third observation
           g3 (float): estimated Lagrange's g function value at third observation
           rho_1_sr (float): estimated slant range at first observation
           rho_2_sr (float): estimated slant range at second observation
           rho_3_sr (float): estimated slant range at third observation
    """
    # get Julian date of observations
    t1 = obs_t[0]
    t2 = obs_t[1]
    t3 = obs_t[2]

    # compute Line-Of-Sight (LOS) vectors
    rho1 = unit_vec(obs_radec[0].ra.rad, obs_radec[0].dec.rad)
    rho2 = unit_vec(obs_radec[1].ra.rad, obs_radec[1].dec.rad)
    rho3 = unit_vec(obs_radec[2].ra.rad, obs_radec[2].dec.rad)

    # compute time differences; make sure time units are consistent!
    tau1 = (t1-t2).jd
    tau3 = (t3-t2).jd
    tau = (tau3-tau1)

    p = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    p[0] = np.cross(rho2, rho3)
    p[1] = np.cross(rho1, rho3)
    p[2] = np.cross(rho1, rho2)

    D0  = np.dot(rho1, p[0])

    D = np.zeros((3,3))

    for i in range(0,3):
        for j in range(0,3):
            D[i,j] = np.dot(R[i], p[j])

    A = (-D[0,1]*(tau3/tau)+D[1,1]+D[2,1]*(tau1/tau))/D0
    B = (D[0,1]*(tau3**2-tau**2)*(tau3/tau)+D[2,1]*(tau**2-tau1**2)*(tau1/tau))/(6*D0)

    E = np.dot(R[1], rho2)
    Rsub2p2 = np.dot(R[1], R[1])

    a = -(A**2+2.0*A*E+Rsub2p2)
    b = -2.0*mu*B*(A+E)
    c = -(mu**2)*(B**2)

    #get all real, positive solutions to the Gauss polynomial
    gauss_poly_coeffs = np.zeros((9,))
    gauss_poly_coeffs[0] = 1.0
    gauss_poly_coeffs[2] = a
    gauss_poly_coeffs[5] = b
    gauss_poly_coeffs[8] = c

    gauss_poly_roots = np.roots(gauss_poly_coeffs)
    rt_indx = np.where( np.isreal(gauss_poly_roots) & (gauss_poly_roots >= 0.0) )
    if len(rt_indx[0]) > 1: #-1:#
        print('WARNING: Gauss polynomial has more than 1 real, positive solution')
        print('Number of solutions = ', len(rt_indx[0]))
        print('Real solutions = ', np.real(gauss_poly_roots[rt_indx[0]]))
        print('r2_root index = ', r2_root_ind)
        print(" ")

    r2_star = np.real(gauss_poly_roots[rt_indx[0][r2_root_ind]])
    print('r2_star = ', r2_star)
    print(" ")

    num1 = 6.0*(D[2,0]*(tau1/tau3) + D[1,0]*(tau/tau3))*(r2_star**3) + mu*D[2,0]*(tau**2-tau1**2)*(tau1/tau3)
    den1 = 6.0*(r2_star**3) + mu*(tau**2 - tau3**2)

    rho_1_sr = ((num1/den1) - D[0,0])/D0

    rho_2_sr = A + (mu*B)/(r2_star**3)

    num3 = 6.0 * (D[0,2] * (tau3/tau1) - D[1,2]*(tau/tau1))*(r2_star**3) + mu*D[0,2]*(tau**2-tau3**2)*(tau3/tau1)
    den3 = 6.0 * (r2_star**3) + mu*(tau**2 - tau1**2)

    rho_3_sr = ((num3/den3) - D[2,2])/D0

    r1 = R[0]+rho_1_sr*rho1
    r2 = R[1]+rho_2_sr*rho2
    r3 = R[2]+rho_3_sr*rho3

    f1 = f_series(mu, r2_star, tau1)
    f3 = f_series(mu, r2_star, tau3)

    g1 = g_series(mu, r2_star, tau1)
    g3 = g_series(mu, r2_star, tau3)

    v2 = gauss_velocity(r1, r2, r3, t1, t2, t3, r2_star=r2_star)

    return r1, r2, r3, v2, D, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_sr, rho_2_sr, rho_3_sr

def get_angle_amb(angle1, angle2):
    def normalize_angle(angle):
        return (angle + 2 * np.pi) % (2 * np.pi)

    angle1_normalized = normalize_angle(angle1)
    angle2_normalized = normalize_angle(angle2)

    # Calculate the absolute differences between angle1 and angle2, and their complements
    diff_1 = abs(angle1_normalized - angle2_normalized)
    diff_2 = abs(2 * np.pi - diff_1)

    # Choose the smaller difference as the corresponding angle
    corresponding_angle_diff = min(diff_1, diff_2)

    # Check if the corresponding angle is within a small tolerance (e.g., 1e-12)
    if corresponding_angle_diff < 1e-12:
        return angle2_normalized

    return angle1_normalized

def orbital_elements(r, rdot):
    """Calculate the orbital elements from the position and velocity vector

       Args:
           r: position of the asteroid
           rdot: velocity of the asteroid

       Returns:
           return the velocity vector at the second observation
    """    

    mod_r = np.sqrt(np.sum(r*r))
    mod_rdot = np.sqrt(np.sum(rdot*rdot))

    # Semi-Major Axis, a
    a = 1/(2/mod_r - (mod_rdot**2)/mu_Sun)

    # Eccentricity of Orbit, e
    prod = np.cross(r,rdot)
    mod_prod = np.sqrt(np.sum(prod*prod))
    e = np.sqrt(1 - (mod_prod**2)/(mu_Sun*a))

    # Inclination of Asteroid Orbit (i)
    l = np.cross(r,rdot)
    irads = np.arctan(np.sqrt(l[0]**2 + l[1]**2)/l[2])
    i = irads*180/np.pi

    #Inclination of Asteroid Orbit (i)
    hz = r[0]*rdot[1] - r[1]*rdot[0]
    h = np.sqrt((r[1]*rdot[2] - r[2]*rdot[1])**2 + (r[2]*rdot[0]-r[0]*rdot[2])**2 + (r[0]*rdot[1]-r[1]*rdot[0])**2)
    i = np.arccos(hz/h) * 180/np.pi

    # Longitude of the ascending Node (o)
    res = np.arctan2(r[1]*rdot[2] - r[2]*rdot[1], r[0]*rdot[2] - r[2]*rdot[0])
    if res >= 0.0:
         o = res*180/np.pi
    else:
         o = (res + 2.0*np.pi)*180/np.pi

    # Longitude of Ascending Node (o)
    mod_l = np.sqrt(np.sum(l*l))
    o1 = np.arcsin(l[0]/(mod_l*np.sin(irads)))
    o2 = np.arccos(-l[1]/(mod_l*np.sin(irads)))
    orads = get_angle_amb(o1,o2)

    # Argument of the Perihilion (w)
    U1 = np.arcsin(r[2]/(mod_r*np.sin(irads)))
    U2 = np.arccos((r[0]*np.cos(orads)+r[1]*np.sin(orads))/mod_r)
    U = get_angle_amb(U1,U2)
    
    v1 = np.arcsin((((a*(1-e**2))/mod_l)*(np.sum(r*rdot)/mod_r))/e)
    v2 = np.arccos(((a*(1-e**2))/mod_r - 1)/e)
    v = get_angle_amb(v1,v2)
    
    wrads = U - v
    w = wrads*180/np.pi % 360
    
    # Mean Anomaly (M)
    E = np.arccos((1-mod_r/a)/e)
    Mrads = E - e*np.sin(E)
    M = Mrads*180/np.pi
    return a, e, i, o, w, M

def gauss_refinement(mu, tau1, tau3, r2, v2, atol, D, R, rho1, rho2, rho3, f_1, g_1, f_3, g_3):
    """Perform refinement of Gauss method.

       Args:
           mu (float): gravitational parameter of center of attraction
           tau1 (float): time interval from second to first observation
           tau3 (float): time interval from second to third observation
           r2 (1x3 array): estimated position at second observation
           v2 (1x3 array): estimated velocity at second observation
           atol (float): absolute tolerance of universal Kepler anomaly computation
           D (3x3 array): auxiliary matrix
           R (1x3 array): three observer position vectors
           rho1 (1x3 array): LOS vector at first observation
           rho2 (1x3 array): LOS vector at second observation
           rho3 (1x3 array): LOS vector at third observation
           f_1 (float): estimated Lagrange's f function value at first observation
           g_1 (float): estimated Lagrange's g function value at first observation
           f_3 (float): estimated Lagrange's f function value at third observation
           g_3 (float): estimated Lagrange's g function value at third observation

       Returns:
           r1 (1x3 array): updated position at first observation
           r2 (1x3 array): updated position at second observation
           r3 (1x3 array): updated position at third observation
           v2 (1x3 array): updated velocity at second observation
           rho_1_sr (float): updated slant range at first observation
           rho_2_sr (float): updated slant range at second observation
           rho_3_sr (float): updated slant range at third observation
           f_1_new (float): updated Lagrange's f function value at first observation
           g_1_new (float): updated Lagrange's g function value at first observation
           f_3_new (float): updated Lagrange's f function value at third observation
           g_3_new (float): updated Lagrange's g function value at third observation
    """
    refinement_success = 1

    xi1 = univkepler(tau1, r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu, iters=10, atol=atol)
    xi3 = univkepler(tau3, r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu, iters=10, atol=atol)

    if np.isnan(xi1) == True or np.isnan(xi3) == True:
        refinement_success = 0
        return np.nan, r2, np.nan, v2, rho1, rho2, rho3, f_1, g_1, f_3, g_3, refinement_success

    r0_ = np.sqrt((r2[0]**2)+(r2[1]**2)+(r2[2]**2))
    v20_ = (v2[0]**2)+(v2[1]**2)+(v2[2]**2)
    alpha0_ = (2.0/r0_)-(v20_/mu)

    z1_ = alpha0_*(xi1**2)
    f_1_new = (f_1+lagrangef_(xi1, z1_, r0_))/2
    g_1_new = (g_1+lagrangeg_(tau1, xi1, z1_, mu))/2

    z3_ = alpha0_*(xi3**2)
    f_3_new = (f_3+lagrangef_(xi3, z3_, r0_))/2
    g_3_new = (g_3+lagrangeg_(tau3, xi3, z3_, mu))/2

    denum = f_1_new*g_3_new-f_3_new*g_1_new
    if np.isinf(np.abs(denum)) == True:
        # one of the terms in denum became really big :(
        refinement_success = 0
        return np.nan, r2, np.nan, v2, rho1, rho2, rho3, f_1, g_1, f_3, g_3, refinement_success

    c1_ = g_3_new/denum
    c3_ = -g_1_new/denum

    D0  = np.dot(rho1, np.cross(rho2, rho3))

    rho_1_sr = (-D[0,0]+D[1,0]/c1_-D[2,0]*(c3_/c1_))/D0
    rho_2_sr = (-c1_*D[0,1]+D[1,1]-c3_*D[2,1])/D0
    rho_3_sr = (-D[0,2]*(c1_/c3_)+D[1,2]/c3_-D[2,2])/D0

    r1 = R[0]+rho_1_sr*rho1
    r2 = R[1]+rho_2_sr*rho2
    r3 = R[2]+rho_3_sr*rho3

    v2 = (-f_3_new*r1+f_1_new*r3)/denum

    return r1, r2, r3, v2, rho_1_sr, rho_2_sr, rho_3_sr, f_1_new, g_1_new, f_3_new, g_3_new, refinement_success

def where_is_my_rock(img1, img2, img3, coord1, coord2, coord3):
    obs_radec = np.array([coord1, coord2, coord3])
    
    t1, loc1, observatory1 = observation(img1)
    t2, loc2, observatory2 = observation(img2)
    t3, loc3, observatory3 = observation(img3)
    obs_t = np.array([t1, t2, t3])

    Earth = Horizons(id='399', location='500@0', epochs=[t1.mjd, t2.mjd, t3.mjd])
    Rvectors = Earth.vectors(refplane="earth") # refplane="earth"

    R1 = np.array([Rvectors["x"][0], Rvectors["y"][0], Rvectors["z"][0]])
    R2 = np.array([Rvectors["x"][1], Rvectors["y"][1], Rvectors["z"][1]])
    R3 = np.array([Rvectors["x"][2], Rvectors["y"][2], Rvectors["z"][2]])

    R = np.array([R1, R2, R3])

    R1dot = np.array([Rvectors["vx"][0], Rvectors["vy"][0], Rvectors["vz"][0]])
    R2dot = np.array([Rvectors["vx"][1], Rvectors["vy"][1], Rvectors["vz"][1]])
    R3dot = np.array([Rvectors["vx"][2], Rvectors["vy"][2], Rvectors["vz"][1]]) 

    return obs_radec, R, obs_t, R1dot, R2dot, R3dot