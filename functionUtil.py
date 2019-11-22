import numpy as np


##### BASIC FUNCTION #######

def shift(solution, shift_number):
    return np.array(solution) - shift_number


def rotate(solution, original_x , rotate_rate=1):
    return solution


def CEC_1(solution=None, problem_size=None, shift=0):
    """
    Rotated High Conditioned Elliptic Function
    x1 = x2 = ... = xn = o
    f(x*) = 100
    """
    res = 0
    constant = np.power(10, 6)
    dim = len(solution)
    for i in range(dim):
        res += np.power(constant, i/dim) * np.square((solution[i] - shift))
    return res


def CEC_2(solution=None, problem_size=None, shift=0):
    """
    Bent cigar function
    f(x*) =  200
    """
    res = 0
    constant = np.power(10, 6)
    dim = len(solution)
    res = np.square((solution[0] - shift))
    for i in range(1, dim):
        res += constant * np.square((solution[i] -  shift))
    return res


def CEC_3(solution=None, problem_size=None, shift=0):
    """
    Discus Function
    f(x*) = 300
    """
    x = solution - shift
    constant = np.power(10, 6)
    dim = len(solution)
    res = constant * np.square(x[0])
    for i in range(1, dim):
        res += np.square(x[i])
    return res


def CEC_4(solution=None, problem_size=None, shift=0):
    """
    rosenbrock Function
    f(x*) = 400
    """
    x = solution - shift
    constant = np.power(10, 6)
    dim = len(solution)
    res = 0
    for i in range(dim - 1):
        res += 100 * np.square(x[i]**2 -  x[i+1]) + np.square(x[i] - 1)
    return res


def CEC_5(solution=None, problem_size=None, shift=0):
    """
    Ackley’s Function
    """
    x = solution - shift
    dim = len(solution)
    res = 0
    A = 0
    B = 0
    A += -0.2 * np.sqrt(np.sum(np.square(x)) /  dim)
    B += np.sum(np.cos(2 * np.pi * x)) / dim
    res = -20 * np.exp(A) - np.exp(B) + 20 + np.e
   # print("res", res)
    return res


def CEC_6(solution=None, problem_size=None, shift=0):
    """
    Weierstrass Function
    """
    x = solution - shift
    dim = len(solution)
    res = 0
    kmax = 1
    a = 0.5
    b = 3
    A = 0
    B = 0
    for i in range(dim):
        for k in range(kmax + 1):
            A += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (x[i] + 0.5))
    for k in range(kmax + 1):
        B += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
    res = A - dim * B
    return res


def CEC_7(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0
    A = np.sum(np.square(x))/4000
    B = 1
    if isinstance(x, np.ndarray):
        dim = len(x)
        for i in range(dim):
            B *= np.cos(x[i]/np.sqrt(i+1))
    else:
        B = np.cos(x)
    res = A - B + 1
    return res


def CEC_8(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0
    dim = len(x)
    res = np.sum(np.square(x)) - 10 * np.sum(np.cos(2 * np.pi * x)) + 10 * dim
    return res


def g9(z, dim):
        if np.abs(z) <= 500:
            return z * np.sin(np.power(np.abs(z), 1/2))
        elif z > 500 :
            return (500 - z % 500) * np.sin(np.sqrt(np.abs(500 - z % 500)))\
                   - np.square(z - 500) / (10000 * dim)
        else:
            return (z % 500 - 500) * np.sin(np.sqrt(np.abs(z % 500 - 500)))\
                   - np.square(z + 500) / (10000 * dim)

def CEC_9(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0
    dim = len(x)
    A = 0
    B = 0
    A = 418.9829 * dim
    z = x + 4.209687462275036e+002
    for i in range(dim):
        B += g9(z[i], dim)
    res = A - B
    return res


def CEC_10(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0
    dim = len(x)
    A = 1
    B = 0
    for i in range(dim):
        temp = 1
        for j in range(32):
            temp += i * (np.abs(np.power(2, j + 1) * x[i]
                    - round(np.power(2, j + 1) * x[i]))) / np.power(2, j)
        A *= np.power(temp, 10 / np.power(dim, 1.2))
    B = 10 / np.square(dim)
    res = B*A - B
    return res


def CEC_11(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0
    dim = len(x)
    A = 0
    B = 0
    A = np.power(np.abs(np.sum(np.square(x)) - dim), 1/4)
    B = (0.5 * np.sum(np.square(x)) + np.sum(x)) / dim
    res = A + B + 0.5
    return res


def CEC_12(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0
    dim = len(x)
    A = 0
    B = 0
    A = np.power(np.abs(np.square(np.sum(np.square(x))) - np.square(np.sum(x))), 1/2)
    B = (0.5 * np.sum(np.square(x)) + np.sum(x)) / dim
    res = A + B + 0.5
    return res


def CEC_13(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0
    dim = len(x)
    A = 0
    B = 0
    for i in range(dim):
        res += CEC_7(CEC_4(x[i : (i + 2) % dim], shift=0), shift=0)
    return res


def CEC_14(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0
    dim = len(x)
    A = 0
    B = 0
    def g(x, y):
        return 0.5 + (np.square(np.sin(np.sqrt(x * x + y * y))) - 0.5) / \
                      np.square(1 + 0.001 * np.square((x*x + y*y)))
    for i in range(dim):
        res += g(x[i], x[(i+1) % dim])
    return res


def C17(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.3 * dim)
    n2 = int(0.3 * dim) + n1
    D = np.arange(dim)

    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_9(x[D[ : n1]]) + CEC_8(x[D[n1 : n2]]) + CEC_1(x[D[n2 : ]]) + 1700 * rate


def C18(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.3 * dim)
    n2 = int(0.3 * dim) + n1
    D = np.arange(dim)
    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_2(x[D[ : n1]]) + CEC_12(x[D[n1 : n2]]) + CEC_8(x[D[n2 : ]]) + 1800 * rate


def C19(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.2 * dim)
    n2 = int(0.2 * dim) + n1
    n3 = int(0.3 * dim) + n2
    D = np.arange(dim)
    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_7(x[D[ : n1]]) + CEC_6(x[D[n1 : n2]]) + CEC_4(x[D[n2 : n3]]) + CEC_14(x[D[n3 : ]]) + 1900 * rate


def C20(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.2 * dim)
    n2 = int(0.2 * dim) + n1
    n3 = int(0.3 * dim) + n2
    D = np.arange(dim)
    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_12(x[D[ : n1]]) + CEC_3(x[D[n1 : n2]]) + CEC_13(x[D[n2 : n3]]) + CEC_8(x[D[n3 : ]]) + 2000 * rate



def slno_f1(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(solution)
    fitness = 0
    for i in range(dim):
        fitness += (10e6)**((i-1)/(dim-1))*(x[i]**2)
    return fitness


def slno_f2(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    fitness = x[0]**2
    for i in range(1, dim):
        fitness += 10e6*x[i]**2
    return fitness


def slno_f3(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    fitness = 10e6*x[0]**2
    for i in range(1, dim):
        fitness += x[i]**2
    return fitness


def slno_f4(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    fitness = 0
    for i in range(dim-1):
        fitness += (100*(x[i]**2 - x[i+1])**2 + (x[i] - 1)**2)
    return fitness


def slno_f5(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    return -20 * np.exp(-0.2 * (1/dim*np.sum([x[i] ** 2 for i in range(dim)])) ** 0.5) - \
           np.exp(1/dim*np.sum([np.cos(2 * np.pi * x[i]) for i in range(dim)])) + 20 + np.e


def slno_f6(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    a = 0.5
    b = 3
    k_max = 20
    return np.sum([np.sum([a**k*np.cos(2*np.pi*b**k*(x[i]+0.5)) for k in range(k_max+1)]) for i in range(dim)]) - \
           dim*np.sum([a**k*np.cos(2*np.pi*b**k*0.5) for k in range(k_max+1)])


def slno_f7(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    return np.sum([1/4000*x[i]**2 for i in range(dim)]) - np.prod([np.cos(x[i]/((i+1)**0.5)) for i in range(dim)]) + 1


def g9_slno(z, dim):
    return g9(z, dim)

def slno_f8(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    return np.sum([x[i]**2 - 10*np.cos(2*np.pi*x[i]) + 10 for i in range(dim)])


def slno_f9(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    z = x + 4.209687462275036e+002
    return 418.9829*dim - np.sum([g9_slno(z[i], dim) for i in range(dim)])


def slno_f10(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    return (10/dim**2)*np.prod([(1+(i+1)*np.sum([(np.abs(2**j*x[i]-round(2**j*x[i]))/2**j)
                                                 for j in range(1, 32)]))**(10/dim**1.2) for i in range(dim)]) - 10/dim**2


def slno_f11(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    return np.abs(np.sum([x[i]**2 for i in range(dim)])-dim)**0.25 + \
            (0.5*np.sum([x[i]**2 for i in range(dim)]) + np.sum([x[i] for i in range(dim)]))/dim + 0.5


def slno_f12(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    return np.abs(np.sum([x[i]**2 for i in range(dim)])**2-np.sum([x[i]**2 for i in range(dim)])**2)**0.5 + \
            (0.5*np.sum([x[i]**2 for i in range(dim)]) + np.sum([x[i] for i in range(dim)]))/dim + 0.5


def slno_f13(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    return CEC_13(solution)


def slno_f14(solution, problem_size=None, shift_num=0):
    x = solution - shift_num
    dim = len(x)
    return CEC_14(solution)

############### MAIN FUNCTIONS ################################
# Unimodal

def islo_uni_F1(solution, problem_size=None, shift_num=1):
    # Sphere function
    x = solution - shift_num
    dim = len(x)
    return np.sum([solution[i]**2 for i in range(dim)])


def islo_uni_F2(solution, problem_size=None, shift_num=1):
    # Schwefel 2.20 Function
    x = solution - shift_num
    dim = len(x)
    return np.sum([np.abs(x[i]) for i in range(dim)])


def islo_uni_F3(solution, problem_size=None, shift_num=1):
    # Schwefel 2.21 Function
    return np.amax(np.abs(solution))


def islo_uni_F4(solution, problem_size=None, shift_num=1):
    # Schwefel 2.22 Function
    x = solution - shift_num
    dim = len(x)
    return np.sum([np.absolute(x[i]) for i in range(dim)]) + np.prod([np.absolute(x[i]) for i in range(dim)])


def islo_uni_F5(solution, problem_size=None, shift_num=1):
    # Sum Squares Function
    x = solution - shift_num
    dim = len(x)
    return np.sum([(i+1)*x[i]**2 for i in range(dim)])


def islo_uni_F6(solution, problem_size=None, shift_num=1):
    # Brown Function
    x = solution - shift_num
    return slno_f7(x)


def islo_uni_F7(solution, problem_size=None, shift_num=1):
    # CEC2015 F1
    x = solution - shift_num
    return slno_f1(x) + 100


def islo_uni_F8(solution, problem_size=None, shift_num=1):
    # CEC 2015 F2
    x = solution - shift_num
    return slno_f2(x) + 200


# Single multimodal
def islo_multi_F9(solution, problem_size=None, shift_num=1):
    # Ackley Function
    x = solution - shift_num
    return slno_f5(x)


def islo_multi_F10(solution, problem_size=None, shift_num=1):
    # Happy Cat Function
    x = solution - shift_num
    return slno_f11(x)


def islo_multi_F11(solution, problem_size=None, shift_num=1):
    # Qing Function
    x = solution - shift_num
    dim = len(x)
    return np.sum([(x[i]**2 - (i+1))**2 for i in range(dim)])


def islo_multi_F12(solution, problem_size=None, shift_num=1):
    # Salomon Function
    x = solution - shift_num
    dim = len(x)
    return 1 - np.cos(2*np.pi*(np.sum(x[i]**2 for i in range(dim)))**0.5) + \
           0.1*(np.sum(x[i]**2 for i in range(dim)))**0.5


def islo_multi_F13(solution, problem_size=None, shift_num=1):
    # Rosenbrock’s Function
    x = solution - shift_num
    return slno_f4(x)

def islo_multi_F14(solution, problem_size=None, shift_num=1):
    # CEC 2015 F3
    x = solution - shift_num
    return slno_f5(x) + 300


def islo_multi_F15(solution, problem_size=None, shift_num=1):
    # CEC 2015 F4
    x = solution - shift_num
    return slno_f1(5.12*x/100) + 400


def islo_multi_F16(solution, problem_size=None, shift_num=1):
    # CEC 2015 F5
    x = solution - shift_num
    return slno_f5(1000*x/100) + 500


# Hybrid
def islo_hybrid_F17(solution, problem_size=None, shift_num=1):
    # CEC 2014 F17
    x = solution - shift_num
    return C17(x)


def islo_hybrid_F18(solution, problem_size=None, shift_num=1):
    # CEC 2014 F18
    x = solution - shift_num
    return C18(x)


def islo_hybrid_F19(solution, problem_size=None, shift_num=1):
    # CEC 2014 F19
    x = solution - shift_num
    return C19(x)


def islo_hybrid_F20(solution, problem_size=None, shift_num=1):
    # CEC 2014 F20
    x = solution - shift_num
    return C20(x)


def islo_hybrid_F21(solution, problem_size=None, shift_num=1):
    # CEC 2015 F6
    x = solution - shift_num
    random_indices = np.arange(len(x))
    np.random.shuffle(random_indices)
    p = [0.3, 0.3, 0.4]
    x1_index = int(p[0]*len(x))
    x2_index = x1_index + int(p[1] * len(x))

    x1 = x[random_indices[0:x1_index]]
    x2 = x[random_indices[x1_index: x2_index]]
    x3 = x[random_indices[x2_index:]]
    # print(len(x1), len(x2), len(x3))
    return slno_f9(x1) + slno_f8(x2) + slno_f1(x3) + 600


def islo_hybrid_F22(solution, problem_size=None, shift_num=1):
    # CEC 2015 F7
    x = solution - shift_num
    random_indices = np.arange(len(x))
    np.random.shuffle(random_indices)
    p = [0.2, 0.2, 0.3, 0.3]
    x1_index = int(p[0] * len(x))
    x2_index = x1_index + int(p[1] * len(x))
    x3_index = x2_index + int(p[2] * len(x))

    x1 = x[random_indices[0:x1_index]]
    x2 = x[random_indices[x1_index: x2_index]]
    x3 = x[random_indices[x2_index:x3_index]]
    x4 = x[random_indices[x3_index:]]
    # print(len(x1), len(x2), len(x3), len(x4))
    return slno_f7(x1) + slno_f6(x2) + slno_f4(x3) + slno_f14(x4) + 700


def islo_hybrid_F23(solution, problem_size=None, shift_num=1):
    # CEC 2015 F8
    x = solution - shift_num
    random_indices = np.arange(len(x))
    np.random.shuffle(random_indices)
    p = [0.1, 0.2, 0.2, 0.2, 0.3]
    x1_index = int(p[0] * len(x))
    x2_index = x1_index + int(p[1] * len(x))
    x3_index = x2_index + int(p[2] * len(x))
    x4_index = x3_index + int(p[3] * len(x))

    x1 = x[random_indices[0:x1_index]]
    x2 = x[random_indices[x1_index: x2_index]]
    x3 = x[random_indices[x2_index:x3_index]]
    x4 = x[random_indices[x3_index:x4_index]]
    x5 = x[random_indices[x4_index:]]
    # print(len(x1), len(x2), len(x3), len(x4))
    return slno_f14(x1) + slno_f12(x2) + slno_f4(x3) + slno_f9(x4) + slno_f1(x5) + 800


# Composition
def islo_compos_F24(solution, problem_size=None, shift_num=1):
    # CEC 2015 F9
    shift_arr = [1, 2, 3]
    sigma = [20, 20, 20]
    lamda = [1, 1, 1]
    bias = [900, 1000, 1100]
    fun = [slno_f9, islo_multi_F15, slno_f12]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = solution - shift_arr[i]
        w[i] = 1 / np.sqrt(np.sum(x**2)) \
               * np.exp(-1 * np.sum(x**2) / (2 * dim * sigma[i]**2))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution) + bias[i])
    return res + 900


def islo_compos_F25(solution, problem_size=None, shift_num=1):
    # CEC 2015 F10
    shift_arr = [1, 2, 3]
    sigma = [10, 30, 50]
    lamda = [1, 1, 1]
    bias = [1000, 1100, 1200]
    fun = [islo_hybrid_F21, islo_hybrid_F22, islo_hybrid_F23]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = solution - shift_arr[i]
        w[i] = 1 / np.sqrt(np.sum(x**2)) \
               * np.exp(-1 * np.sum(x**2) / (2 * dim * sigma[i]**2))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution) + bias[i])
    return res + 1000


def islo_compos_F26(solution, problem_size=None, shift_num=1):
    # CEC 2015 F11
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 10, 10, 20, 20]
    lamda = [10, 10, 2.5, 25,1e-6]
    bias = [1100, 1200, 1300, 1400, 1500]
    fun = [slno_f12, slno_f8, slno_f9, slno_f6, islo_uni_F7]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = solution - shift_arr[i]
        w[i] = 1 / np.sqrt(np.sum(x**2)) \
               * np.exp(-1 * np.sum(x**2) / (2 * dim * sigma[i]**2))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution) + bias[i])
    return res + 1100


def islo_compos_F27(solution, problem_size=None, shift_num=1):
    # CEC 2015 F12
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 10, 20, 20, 30]
    lamda = [0.25, 1, 1e-7, 10, 10]
    bias = [1200, 1300, 1400, 1500, 1600]
    fun = [slno_f9, slno_f8, islo_uni_F7, slno_f14, slno_f11]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = solution - shift_arr[i]
        w[i] = 1 / np.sqrt(np.sum(x**2)) \
               * np.exp(-1 * np.sum(x**2) / (2 * dim * sigma[i]**2))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution) + bias[i])
    return res + 1200


def islo_compos_F28(solution, problem_size=None, shift_num=1):
    # CEC 2015 F13
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 10, 10, 20, 20]
    lamda = [1, 10, 1, 25, 10]
    bias = [1300, 1400, 1500, 1600, 1700]
    fun = [islo_hybrid_F23, slno_f8, islo_hybrid_F21, slno_f9, slno_f14]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = solution - shift_arr[i]
        w[i] = 1 / np.sqrt(np.sum(x**2)) \
               * np.exp(-1 * np.sum(x**2) / (2 * dim * sigma[i]**2))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution) + bias[i])
    return res + 1300


def islo_compos_F29(solution, problem_size=None, shift_num=1):
    # CEC 2015 F14
    shift_arr = [1, 2, 3, 4, 5, 6, 7]
    sigma = [10, 20, 30, 40, 50, 50, 50]
    lamda = [10,2.5, 2.5, 10,1e-6,1e-6, 10]
    bias = [1400, 1500, 1600, 1700, 1800, 1900, 2000]
    fun = [slno_f11, slno_f13, slno_f9, slno_f14, islo_uni_F7, slno_f2, slno_f8]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = solution - shift_arr[i]
        w[i] = 1 / np.sqrt(np.sum(x**2)) \
               * np.exp(-1 * np.sum(x**2) / (2 * dim * sigma[i]**2))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution) + bias[i])
    return res + 1400


def islo_compos_F30(solution, problem_size=None, shift_num=1):
    # CEC 2015 F15
    shift_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sigma = [10, 10, 20, 20, 30, 30, 40, 40, 50, 50]
    lamda = [0.1,2.5e-1, 0.1, 2.5e-2, 1e-3, 0.1, 1e-5, 10, 2.5e-2, 1e-3]
    bias = [1500, 1600, 1600, 1700, 1700, 1800, 1800, 1900, 1900, 2000]
    fun = [slno_f8, slno_f6, slno_f11, slno_f9, islo_multi_F15, slno_f12, slno_f5, slno_f14, slno_f13, slno_f5]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = solution - shift_arr[i]
        w[i] = 1 / np.sqrt(np.sum(x**2)) \
               * np.exp(-1 * np.sum(x**2) / (2 * dim * sigma[i]**2))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution) + bias[i])
    return res + 1400
