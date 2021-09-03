import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=12) 

def dynamics(x, u):
    m = 1
    k = 100
    h = 0.1
    if (u > 1):
        w1 = m / (m + h ** 2.0 * k)
        w2 = h ** 2.0 * k / (m + h **2.0 * k)
        return np.array([w1 * 1.0 + w2 * u, w1 * 1.0 + w2 * u])
    else:
        return np.array([u, 1.0])

w_lst = np.random.normal(0, 1.0, 1000)
u_lst = np.linspace(-2, 2, 1000)
xnext_lst = np.zeros((1000,2))
xtrue_lst = np.zeros((1000,2))

for i in range(len(u_lst)):
    bundle = np.array([0.0, 0.0])
    for j in range(len(w_lst)):
        bundle += dynamics(0.0, u_lst[i] + w_lst[j])
    bundle /= len(w_lst)
    xnext_lst[i] = bundle
    xtrue_lst[i] = dynamics(0.0, u_lst[i])

fig = plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(u_lst, xnext_lst[:,1], 'r--', label='bundled dynamics')
plt.plot(u_lst, xtrue_lst[:,1], 'r-', label='true dynamics')
plt.legend()
plt.xlabel('u^1')
plt.ylabel('x_{k+1}^2')

def dynamics_anitescu(phi):
    if phi > 0.02:
        return 0
    if phi <= 0.02 and phi > 0:
        return -0.09 / 0.02 * phi + 0.09
    if phi < 0:
        return 0.09

def dynamics_lcp(phi):
    if phi > 0:
        return 0
    else:
        return 0.09

w_lst = np.random.normal(0, 0.01, 4000)
phi_lst = np.linspace(0.0, 0.1, 1000)

anitescu_lst = []
anitescu_bundle_lst = []
lcp_lst = []
lcp_bundle_lst = []

for i in range(len(phi_lst)):
    anitescu_lst.append(dynamics_anitescu(phi_lst[i]))
    lcp_lst.append(dynamics_lcp(phi_lst[i]))    
    bundle_anitescu = 0.0
    bundle_lcp = 0.0    
    for j in range(len(w_lst)):
        bundle_anitescu += dynamics_anitescu(phi_lst[i] + w_lst[j])
        bundle_lcp += dynamics_lcp(phi_lst[i] + w_lst[j])
    bundle_anitescu /= len(w_lst)
    bundle_lcp /= len(w_lst)
    anitescu_bundle_lst.append(bundle_anitescu)
    lcp_bundle_lst.append(bundle_lcp)
    

plt.subplot(1,2,2)
plt.plot(np.array(lcp_lst), phi_lst, '-', color='blue', 
    label='LCP Dynamics')
plt.plot(np.array(lcp_bundle_lst), phi_lst, '--', color='blue',
    label="Bundle Dynamics (LCP)")
plt.plot(np.array(anitescu_lst), phi_lst, '-', color='springgreen',
    label="Anitescu")
plt.plot(np.array(anitescu_bundle_lst), phi_lst, '--', color='springgreen',
    label="Bundle Dynamics (Anitescu)")

plt.xlabel('x_{k+1}^2')
plt.ylabel('phi')
plt.legend()
plt.show()

image_format = "svg"
image_name = "bundle_dynamics.svg"
fig.savefig(image_name, format=image_format, dpi=1200)
